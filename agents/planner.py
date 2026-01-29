"""Planner Agent for generating domain-specific research queries.

This agent analyzes product metadata extracted by the ClarificationAgent and
generates targeted research queries for comprehensive market research.
"""

import json
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from agents.base_agent import BaseAgent
from app.logger import get_logger
from app.state import Query, State, Task


class QueryItem(BaseModel):
    """A single research query with metadata."""

    text: str = Field(
        description="The search query text"
    )
    category: str = Field(
        description="Query category: competitor, pain_points, workflow, or compliance"
    )
    priority: str = Field(
        description="Query priority: high, medium, or low"
    )
    expected_sources: List[str] = Field(
        description="Expected source types for this query",
        default_factory=list
    )


class QueryGenerationResponse(BaseModel):
    """Structured response from the planner agent."""

    queries: List[QueryItem] = Field(
        description="List of generated research queries"
    )
    rationale: str = Field(
        description="Explanation of why these queries were chosen"
    )


class QueryValidationError(Exception):
    """Raised when query validation fails."""
    pass


class PlannerAgent(BaseAgent):
    """Agent responsible for generating domain-specific research queries.

    This agent:
    1. Reads metadata extracted by ClarificationAgent
    2. Generates 15-20 targeted research queries
    3. Validates query count, category distribution, and uniqueness
    4. Updates state.research_plan with Query objects
    5. Updates task_board with planning status

    The agent uses OpenAI's structured output mode for reliable query generation.
    """

    # Category distribution requirements
    CATEGORY_REQUIREMENTS = {
        "competitor": (5, 7),      # min, max
        "pain_points": (5, 7),
        "workflow": (3, 4),
        "compliance": (2, 3),
    }

    # Valid expected sources
    VALID_SOURCES = {
        "comparison_sites", "pricing_pages", "forums", "reviews",
        "articles", "docs", "reports"
    }

    # Fuzzy matching threshold for duplicate detection
    DUPLICATE_THRESHOLD = 85  # Similarity score above this is considered duplicate

    def __init__(self, name: str, client: OpenAI) -> None:
        """Initialize the planner agent.

        Args:
            name: Agent identifier (typically "planner")
            client: Configured OpenAI client instance
        """
        super().__init__(name, client)
        self.logger = get_logger(__name__)

    def run(self, state: State) -> State:
        """Execute the planner agent's ReAct loop.

        Args:
            state: Current shared state with metadata populated

        Returns:
            Updated state with research_plan populated

        Raises:
            ValueError: If metadata is not available or invalid
            QueryValidationError: If generated queries fail validation
        """
        self.logger.info("Starting planner agent")

        # 1. Think: Check prerequisites
        if not self._check_prerequisites(state):
            return state

        # Check if planning already done
        if state.research_plan.queries:
            self.logger.info("Research plan already exists, skipping")
            self._log_action(state, "Skipped - research plan already exists")
            return state

        # Add task to task board
        task = Task(
            id=f"T-PLAN-{state.run_id[:8]}",
            owner="planner",
            status="doing",
            description="Generate domain-specific research queries"
        )
        state.task_board.append(task)

        try:
            # 2. Act: Load prompt and call LLM
            self._log_action(state, "Loading planner prompt")
            prompt_template = self._load_prompt()

            # Build metadata context
            metadata_json = self._build_metadata_context(state)
            prompt = prompt_template.replace("{{metadata}}", metadata_json)

            self.logger.info("Calling LLM for query generation")
            self._log_action(
                state,
                "Calling LLM with structured output mode",
                details={
                    "model": self.config.openai_model,
                    "domain": state.metadata.domain,
                    "target_user": state.metadata.target_user
                }
            )

            # Build JSON schema for structured output
            json_schema = self._build_json_schema()

            messages = [
                {
                    "role": "system",
                    "content": "You are a research strategist generating targeted search queries for product research."
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
                temperature=0.5  # Moderate temperature for creative but consistent queries
            )

            # 3. Observe: Parse and validate response
            self._log_action(state, "Parsing LLM response")
            observations = self._observe(llm_response)

            if not observations.get("valid"):
                raise ValueError(f"Invalid LLM response: {observations.get('error')}")

            query_response: QueryGenerationResponse = observations["data"]

            # 4. Validate queries
            self._log_action(state, "Validating generated queries")
            validation_result = self._validate_queries(query_response.queries)

            if not validation_result["valid"]:
                self.logger.warning(
                    f"Query validation issues: {validation_result['issues']}"
                )
                self._log_action(
                    state,
                    "Query validation warnings",
                    details={"issues": validation_result["issues"]}
                )

                # Attempt to fix minor issues
                if validation_result.get("can_proceed"):
                    self.logger.info("Proceeding with minor validation issues")
                else:
                    raise QueryValidationError(
                        f"Query validation failed: {validation_result['issues']}"
                    )

            # Remove duplicates if any
            deduplicated_queries = self._remove_duplicates(query_response.queries)

            # 5. Update: Modify state with generated queries
            self._log_action(
                state,
                "Updating state with research plan",
                details={
                    "query_count": len(deduplicated_queries),
                    "rationale": query_response.rationale[:100] + "..."
                }
            )

            # Convert QueryItems to Query objects
            for query_item in deduplicated_queries:
                query = Query(
                    id=f"Q-{uuid.uuid4().hex[:8]}",
                    text=query_item.text,
                    category=query_item.category,  # type: ignore
                    priority=query_item.priority,  # type: ignore
                    status="pending",
                    expected_sources=query_item.expected_sources
                )
                state.research_plan.queries.append(query)

            # Log distribution
            distribution = self._get_category_distribution(deduplicated_queries)
            self._log_action(
                state,
                "Query distribution",
                details={"distribution": distribution}
            )

            # Mark planning task as done
            for t in state.task_board:
                if t.id == task.id:
                    t.status = "done"
                    break

            # Add research task to task board
            research_task = Task(
                id=f"T-RESEARCH-{state.run_id[:8]}",
                owner="researcher",
                status="pending",
                description=f"Execute {len(state.research_plan.queries)} research queries"
            )
            state.task_board.append(research_task)

            # 6. Reflect: Log completion
            self._log_action(
                state,
                "Planning completed successfully",
                details={
                    "total_queries": len(state.research_plan.queries),
                    "high_priority": len([q for q in state.research_plan.queries if q.priority == "high"]),
                    "medium_priority": len([q for q in state.research_plan.queries if q.priority == "medium"]),
                    "low_priority": len([q for q in state.research_plan.queries if q.priority == "low"])
                }
            )

            self.logger.info(
                f"Planning completed: generated {len(state.research_plan.queries)} queries"
            )

            return state

        except Exception as e:
            # Mark task as blocked
            for t in state.task_board:
                if t.id == task.id:
                    t.status = "blocked"
                    break

            self.logger.error(f"Planning failed: {e}")
            self._log_action(state, f"Planning failed: {str(e)}")
            raise

    def _check_prerequisites(self, state: State) -> bool:
        """Check if prerequisites for planning are met.

        Args:
            state: Current state

        Returns:
            True if prerequisites are met, False otherwise
        """
        # Check if clarification has been done
        if not state.metadata.domain:
            self.logger.error("Metadata domain not set - clarification not complete")
            self._log_action(
                state,
                "Prerequisites not met - missing metadata",
                details={"missing": "domain"}
            )
            return False

        if not state.metadata.target_user:
            self.logger.error("Target user not set - clarification not complete")
            self._log_action(
                state,
                "Prerequisites not met - missing metadata",
                details={"missing": "target_user"}
            )
            return False

        return True

    def _build_metadata_context(self, state: State) -> str:
        """Build JSON string of metadata for the prompt.

        Args:
            state: Current state

        Returns:
            JSON string representation of metadata
        """
        metadata_dict = {
            "domain": state.metadata.domain,
            "industry_tags": state.metadata.industry_tags,
            "target_user": state.metadata.target_user,
            "geography": state.metadata.geography,
            "compliance_contexts": state.metadata.compliance_contexts
        }

        return json.dumps(metadata_dict, indent=2)

    def _build_json_schema(self) -> Dict[str, Any]:
        """Build the JSON schema for OpenAI structured output.

        Returns:
            JSON schema dictionary compatible with OpenAI API
        """
        return {
            "name": "query_generation_response",
            "description": "Research queries generated for product research",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "List of research queries",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The search query text"
                                },
                                "category": {
                                    "type": "string",
                                    "enum": ["competitor", "pain_points", "workflow", "compliance"],
                                    "description": "Query category"
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Query priority"
                                },
                                "expected_sources": {
                                    "type": "array",
                                    "description": "Expected source types",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["text", "category", "priority", "expected_sources"],
                            "additionalProperties": False
                        }
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why these queries were chosen"
                    }
                },
                "required": ["queries", "rationale"],
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
            query_response = QueryGenerationResponse(**data_dict)

            self.logger.debug(
                f"Successfully parsed {len(query_response.queries)} queries"
            )

            return {
                "valid": True,
                "data": query_response
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

    def _validate_queries(self, queries: List[QueryItem]) -> Dict[str, Any]:
        """Validate the generated queries.

        Checks:
        - Total count: 15-20
        - Category distribution within ranges
        - No near-duplicates
        - Valid expected sources

        Args:
            queries: List of generated query items

        Returns:
            Dictionary with validation result and issues
        """
        issues = []
        can_proceed = True

        # Check total count
        total_count = len(queries)
        if total_count < 15:
            issues.append(f"Too few queries: {total_count} (need 15-20)")
            can_proceed = total_count >= 12  # Allow proceeding with 12+
        elif total_count > 20:
            issues.append(f"Too many queries: {total_count} (need 15-20)")
            can_proceed = True  # Can proceed, will trim if needed

        # Check category distribution
        distribution = self._get_category_distribution(queries)

        for category, (min_count, max_count) in self.CATEGORY_REQUIREMENTS.items():
            count = distribution.get(category, 0)

            # For compliance, allow 0 if no compliance contexts
            if category == "compliance" and count == 0:
                continue

            if count < min_count:
                issues.append(f"{category}: {count} queries (need {min_count}-{max_count})")
                if count < min_count - 1:
                    can_proceed = False
            elif count > max_count:
                issues.append(f"{category}: {count} queries (need {min_count}-{max_count})")
                # Can proceed with extra queries

        # Check for duplicates
        duplicates = self._find_duplicates(queries)
        if duplicates:
            issues.append(f"Found {len(duplicates)} near-duplicate queries")
            can_proceed = True  # Can proceed, will remove duplicates

        # Validate expected sources
        invalid_sources = []
        for query in queries:
            for source in query.expected_sources:
                if source not in self.VALID_SOURCES:
                    invalid_sources.append(source)

        if invalid_sources:
            unique_invalid = list(set(invalid_sources))
            issues.append(f"Invalid sources: {unique_invalid}")
            can_proceed = True  # Can proceed with unknown sources

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "can_proceed": can_proceed
        }

    def _get_category_distribution(self, queries: List[QueryItem]) -> Dict[str, int]:
        """Get the distribution of queries by category.

        Args:
            queries: List of query items

        Returns:
            Dictionary mapping category to count
        """
        return dict(Counter(q.category for q in queries))

    def _find_duplicates(self, queries: List[QueryItem]) -> List[Tuple[int, int, float]]:
        """Find near-duplicate queries using fuzzy matching.

        Args:
            queries: List of query items

        Returns:
            List of tuples (index1, index2, similarity_score) for duplicates
        """
        duplicates = []

        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                similarity = fuzz.ratio(
                    queries[i].text.lower(),
                    queries[j].text.lower()
                )
                if similarity >= self.DUPLICATE_THRESHOLD:
                    duplicates.append((i, j, similarity))

        return duplicates

    def _remove_duplicates(self, queries: List[QueryItem]) -> List[QueryItem]:
        """Remove near-duplicate queries, keeping the first occurrence.

        Args:
            queries: List of query items

        Returns:
            Deduplicated list of query items
        """
        duplicates = self._find_duplicates(queries)

        if not duplicates:
            return queries

        # Get indices to remove (keep first occurrence)
        indices_to_remove = set()
        for i, j, _ in duplicates:
            indices_to_remove.add(j)

        self.logger.info(f"Removing {len(indices_to_remove)} duplicate queries")

        return [q for idx, q in enumerate(queries) if idx not in indices_to_remove]
