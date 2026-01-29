"""Clarification Agent for extracting structured metadata from product ideas.

This agent analyzes the raw product idea and extracts structured information
including domain, industry tags, target users, and compliance requirements.
"""

import json
from typing import Any, Dict, List

from openai import OpenAI
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from app.logger import get_logger
from app.state import State, Task


class ClarificationResponse(BaseModel):
    """Structured response from the clarification agent.

    This model defines the expected output format when analyzing a product idea.
    """

    domain: str = Field(
        description="Primary domain/category of the product"
    )
    industry_tags: List[str] = Field(
        description="2-4 specific industry tags",
        min_length=2,
        max_length=4
    )
    target_user: str = Field(
        description="Specific target user description"
    )
    geography: str = Field(
        description="Geographic focus (e.g., 'US', 'EU', 'global')"
    )
    compliance_contexts: List[str] = Field(
        description="Relevant compliance/regulatory contexts",
        default_factory=list
    )
    assumptions: List[str] = Field(
        description="Assumptions made during analysis",
        default_factory=list
    )
    clarification_questions: List[str] = Field(
        description="2-3 questions to clarify requirements",
        min_length=0,
        max_length=5
    )


class ClarificationAgent(BaseAgent):
    """Agent responsible for extracting structured metadata from raw product ideas.

    This agent:
    1. Analyzes the user's raw product idea
    2. Extracts domain, industry tags, target users, and compliance contexts
    3. Identifies assumptions being made
    4. Generates clarification questions for ambiguous areas
    5. Updates the state metadata with extracted information

    The agent uses OpenAI's structured output mode to ensure reliable parsing.
    """

    def __init__(self, name: str, client: OpenAI) -> None:
        """Initialize the clarification agent.

        Args:
            name: Agent identifier (typically "clarification")
            client: Configured OpenAI client instance
        """
        super().__init__(name, client)
        self.logger = get_logger(__name__)

    def run(self, state: State) -> State:
        """Execute the clarification agent's ReAct loop.

        Args:
            state: Current shared state

        Returns:
            Updated state with metadata populated

        Raises:
            Exception: If clarification fails after retries
        """
        self.logger.info("Starting clarification agent")

        # 1. Think: Check if clarification is needed
        # Skip if already run (either confirmed or pending with questions)
        if state.metadata.clarification_status in ["confirmed", "pending"]:
            # Check if we've already extracted metadata
            if state.metadata.domain:  # Domain is populated means we already ran
                self.logger.info(f"Clarification already done (status: {state.metadata.clarification_status}), skipping")
                self._log_action(state, f"Skipped - clarification already {state.metadata.clarification_status}")
                return state

        # Add task to task board
        task = Task(
            id=f"T-CLARIFY-{state.run_id[:8]}",
            owner="clarification",
            status="doing",
            description="Extract structured metadata from product idea"
        )
        state.task_board.append(task)

        try:
            # 2. Act: Load prompt and call LLM
            self._log_action(state, "Loading clarification prompt")
            prompt_template = self._load_prompt()

            # Replace placeholder with actual idea
            prompt = prompt_template.replace("{{raw_idea}}", state.metadata.raw_idea)

            self.logger.info("Calling LLM for metadata extraction")
            self._log_action(
                state,
                "Calling LLM with structured output mode",
                details={"model": self.config.openai_model}
            )

            # Build JSON schema for structured output
            json_schema = self._build_json_schema()

            messages = [
                {
                    "role": "system",
                    "content": "You are a product analyst expert at extracting structured metadata from product ideas."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Call LLM with structured output
            llm_response = self._call_llm(
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                },
                temperature=0.3  # Lower temperature for more consistent extraction
            )

            # 3. Observe: Parse and validate response
            self._log_action(state, "Parsing LLM response")
            observations = self._observe(llm_response)

            if not observations.get("valid"):
                raise ValueError(f"Invalid LLM response: {observations.get('error')}")

            clarification_data = observations["data"]

            # 4. Update: Modify state with extracted metadata
            self._log_action(
                state,
                "Updating state metadata",
                details={
                    "domain": clarification_data.domain,
                    "target_user": clarification_data.target_user,
                    "num_questions": len(clarification_data.clarification_questions)
                }
            )

            state.metadata.domain = clarification_data.domain
            state.metadata.industry_tags = clarification_data.industry_tags
            state.metadata.target_user = clarification_data.target_user
            state.metadata.geography = clarification_data.geography
            state.metadata.compliance_contexts = clarification_data.compliance_contexts

            # Store assumptions and questions in agent trace for transparency
            if clarification_data.assumptions:
                self._log_action(
                    state,
                    f"Made {len(clarification_data.assumptions)} assumptions",
                    details={"assumptions": clarification_data.assumptions}
                )

            if clarification_data.clarification_questions:
                self._log_action(
                    state,
                    f"Generated {len(clarification_data.clarification_questions)} clarification questions",
                    details={"questions": clarification_data.clarification_questions}
                )
                # Set status to pending if we have questions
                state.metadata.clarification_status = "pending"
            else:
                # No questions means we're confident
                state.metadata.clarification_status = "confirmed"

            # Mark task as done
            for t in state.task_board:
                if t.id == task.id:
                    t.status = "done"
                    break

            # 5. Reflect: Log completion
            self._log_action(
                state,
                "Clarification completed successfully",
                details={
                    "status": state.metadata.clarification_status,
                    "domain": state.metadata.domain,
                    "industry_tags": state.metadata.industry_tags
                }
            )

            self.logger.info(
                f"Clarification completed: domain={state.metadata.domain}, "
                f"target_user={state.metadata.target_user}"
            )

            return state

        except Exception as e:
            # Mark task as blocked
            for t in state.task_board:
                if t.id == task.id:
                    t.status = "blocked"
                    break

            self.logger.error(f"Clarification failed: {e}")
            self._log_action(state, f"Clarification failed: {str(e)}")
            raise

    def _build_json_schema(self) -> Dict[str, Any]:
        """Build the JSON schema for OpenAI structured output.

        Returns:
            JSON schema dictionary compatible with OpenAI API
        """
        return {
            "name": "clarification_response",
            "description": "Structured metadata extracted from a product idea",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Primary domain/category of the product"
                    },
                    "industry_tags": {
                        "type": "array",
                        "description": "2-4 specific industry tags",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4
                    },
                    "target_user": {
                        "type": "string",
                        "description": "Specific target user description"
                    },
                    "geography": {
                        "type": "string",
                        "description": "Geographic focus"
                    },
                    "compliance_contexts": {
                        "type": "array",
                        "description": "Relevant compliance/regulatory contexts",
                        "items": {"type": "string"}
                    },
                    "assumptions": {
                        "type": "array",
                        "description": "Assumptions made during analysis",
                        "items": {"type": "string"}
                    },
                    "clarification_questions": {
                        "type": "array",
                        "description": "2-3 questions to clarify requirements",
                        "items": {"type": "string"},
                        "maxItems": 5
                    }
                },
                "required": [
                    "domain",
                    "industry_tags",
                    "target_user",
                    "geography",
                    "compliance_contexts",
                    "assumptions",
                    "clarification_questions"
                ],
                "additionalProperties": False
            }
        }

    def _observe(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the LLM response.

        Args:
            llm_response: Raw response from _call_llm

        Returns:
            Dictionary with validation status and parsed data
        """
        try:
            content = llm_response.get("content", "")

            if not content:
                return {
                    "valid": False,
                    "error": "Empty response from LLM"
                }

            # Parse JSON response
            data_dict = json.loads(content)

            # Validate with Pydantic
            clarification_data = ClarificationResponse(**data_dict)

            self.logger.debug(f"Successfully parsed clarification response: {clarification_data.domain}")

            return {
                "valid": True,
                "data": clarification_data
            }

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return {
                "valid": False,
                "error": f"Invalid JSON: {e}"
            }

        except Exception as e:
            self.logger.error(f"Failed to validate response: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {e}"
            }
