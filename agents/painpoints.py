"""PainPointsAgent - extracts and clusters pain points from evidence.

This agent analyzes evidence (especially forums, reviews) to identify
and cluster user pain points using LLM-based semantic clustering.
"""

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from app.logger import get_logger
from app.state import PainPoint, State, Task
from rich.console import Console

logger = get_logger(__name__)
console = Console()


class ExtractedPainPoint(BaseModel):
    """Structured pain point extracted by LLM."""

    cluster_name: str = Field(
        description="Short, specific name for the pain point cluster"
    )
    who: str = Field(
        description="Which user segment experiences this pain point"
    )
    what: str = Field(
        description="Concrete description of the pain point with details"
    )
    why: str = Field(
        description="Root cause or reason this is painful"
    )
    severity: str = Field(
        description="Severity level: high, medium, or low"
    )
    frequency: str = Field(
        description="How often this was mentioned in evidence"
    )
    example_quotes: List[str] = Field(
        description="2-3 direct quotes from evidence",
        min_length=1,
        max_length=5
    )


class PainPointsResponse(BaseModel):
    """LLM response format for pain point extraction."""

    pain_points: List[ExtractedPainPoint] = Field(
        description="List of clustered pain points"
    )
    rationale: str = Field(
        description="Brief explanation of clustering approach"
    )


class PainPointsAgent(BaseAgent):
    """Extract and cluster pain points from evidence.

    Process:
    1. Filter evidence for pain point sources (forums, reviews, complaints)
    2. Extract raw pain points using LLM
    3. Cluster similar pain points semantically
    4. Rank by severity (frequency + source credibility)
    5. Link back to evidence IDs
    6. Update state.insights.pain_points

    Attributes:
        name: Agent identifier ("painpoints")
        client: OpenAI client for API calls
    """

    # Evidence types relevant for pain point extraction
    RELEVANT_EVIDENCE_TYPES = {"forum", "review"}

    # Tags that indicate pain point relevance
    RELEVANT_TAGS = {"pain_points", "complaints", "feedback", "issues"}

    # Maximum evidence items to process (to fit in context)
    MAX_EVIDENCE_ITEMS = 30

    # Severity mapping for normalization
    SEVERITY_MAP = {
        "high": "critical",
        "medium": "major",
        "low": "minor"
    }

    def __init__(self, name: str, client: OpenAI) -> None:
        """Initialize the PainPointsAgent.

        Args:
            name: Agent identifier (typically "painpoints")
            client: Configured OpenAI client instance
        """
        super().__init__(name, client)
        self.logger = get_logger(__name__)

    def run(self, state: State) -> State:
        """Execute pain point extraction and clustering.

        Args:
            state: Current shared state containing evidence

        Returns:
            Updated state with pain points added to insights

        Raises:
            Exception: If extraction fails after retries
        """
        self.logger.info("Starting pain point analysis")
        self._log_action(state, "started_painpoint_analysis")

        # Create task on task board
        task = Task(
            id=f"T-PAINPOINTS-{state.run_id[:8]}",
            owner="painpoints",
            status="doing",
            description="Extract and cluster pain points from evidence"
        )
        state.task_board.append(task)

        try:
            # 1. Think: Filter relevant evidence
            relevant_evidence = self._filter_evidence(state.evidence)

            if not relevant_evidence:
                self.logger.warning("No relevant evidence for pain point extraction")
                console.print("[yellow]No forum/review evidence found for pain points[/yellow]")
                self._log_action(
                    state,
                    "no_relevant_evidence",
                    details={"total_evidence": len(state.evidence)}
                )
                # Mark task as done (nothing to do)
                self._mark_task_done(state, task.id)
                return state

            self._log_action(
                state,
                "filtered_evidence",
                details={
                    "total_evidence": len(state.evidence),
                    "relevant_evidence": len(relevant_evidence)
                }
            )

            console.print(
                f"\n[bold cyan]Analyzing {len(relevant_evidence)} sources "
                f"for pain points...[/bold cyan]\n"
            )

            # 2. Act: Extract and cluster pain points using LLM
            pain_points = self._extract_and_cluster(relevant_evidence, state)

            if not pain_points:
                self.logger.warning("No pain points extracted from evidence")
                console.print("[yellow]No pain points identified[/yellow]")
                self._mark_task_done(state, task.id)
                return state

            # 3. Observe & Update: Add pain points to state
            for i, pp in enumerate(pain_points):
                # Link pain point to evidence
                evidence_ids = self._link_evidence(pp, relevant_evidence)

                # Map severity to state model format
                severity = self.SEVERITY_MAP.get(pp.severity.lower(), "major")

                # Create PainPoint for state
                pain_point = PainPoint(
                    id=f"PP{len(state.insights.pain_points) + 1}",
                    cluster_name=pp.cluster_name,
                    who=pp.who,
                    what=pp.what,
                    why=pp.why,
                    description=f"{pp.what} - {pp.why}",  # Combined description
                    severity=severity,
                    frequency=pp.frequency,
                    example_quotes=pp.example_quotes,
                    evidence_ids=evidence_ids
                )

                state.insights.pain_points.append(pain_point)

                self.logger.debug(
                    f"Added pain point: {pp.cluster_name} "
                    f"(severity={pp.severity}, linked to {len(evidence_ids)} sources)"
                )

            # 4. Reflect: Log completion
            console.print(
                f"[green]Identified {len(pain_points)} pain point clusters[/green]\n"
            )

            self._log_action(
                state,
                "completed_painpoint_analysis",
                details={
                    "pain_points_found": len(pain_points),
                    "clusters": [pp.cluster_name for pp in pain_points]
                }
            )

            # Mark task as done
            self._mark_task_done(state, task.id)

            self.logger.info(f"Pain point analysis complete: {len(pain_points)} clusters")

            return state

        except Exception as e:
            # Mark task as blocked on error
            self._mark_task_blocked(state, task.id)
            self.logger.error(f"Pain point extraction failed: {e}")
            self._log_action(state, f"painpoint_extraction_failed: {str(e)}")
            raise

    def _filter_evidence(self, evidence: List[Any]) -> List[Dict[str, Any]]:
        """Filter evidence for pain point relevant sources.

        Args:
            evidence: List of Evidence objects or dicts from state

        Returns:
            List of relevant evidence as dictionaries
        """
        filtered = []

        for e in evidence:
            # Convert to dict if needed
            if hasattr(e, 'model_dump'):
                e_dict = e.model_dump()
            elif hasattr(e, '__dict__'):
                e_dict = dict(e.__dict__)
            else:
                e_dict = dict(e) if isinstance(e, dict) else {}

            # Check if type is relevant
            e_type = e_dict.get("type", "")
            if e_type in self.RELEVANT_EVIDENCE_TYPES:
                filtered.append(e_dict)
                continue

            # Check if any relevant tags present
            e_tags = set(e_dict.get("tags", []))
            if e_tags & self.RELEVANT_TAGS:
                filtered.append(e_dict)
                continue

        # Sort by credibility (high first) and relevance score
        def sort_key(e):
            cred_order = {"high": 0, "medium": 1, "low": 2}
            cred = cred_order.get(e.get("credibility", "low"), 2)
            relevance = e.get("relevance_score", 0.5)
            return (cred, -relevance)

        filtered.sort(key=sort_key)

        self.logger.debug(
            f"Filtered {len(filtered)} relevant evidence items "
            f"from {len(evidence)} total"
        )

        return filtered

    def _extract_and_cluster(
        self,
        evidence: List[Dict[str, Any]],
        state: State
    ) -> List[ExtractedPainPoint]:
        """Use LLM to extract and cluster pain points.

        Args:
            evidence: Filtered relevant evidence
            state: Current state for context

        Returns:
            List of ExtractedPainPoint objects
        """
        # Load prompt template
        try:
            prompt_template = self._load_prompt()
        except FileNotFoundError:
            self.logger.error("Pain points prompt file not found")
            raise

        # Prepare evidence context (limit to max items)
        evidence_context = self._prepare_evidence_context(
            evidence[:self.MAX_EVIDENCE_ITEMS]
        )

        # Build prompt with context
        prompt = prompt_template.replace("{{evidence}}", evidence_context)
        prompt = prompt.replace("{{target_user}}", state.metadata.target_user or "users")
        prompt = prompt.replace("{{domain}}", state.metadata.domain or "general")

        # Build messages for LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a product analyst extracting user pain points from research. "
                    "Analyze the evidence carefully and identify distinct pain point clusters."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Build JSON schema for structured output
        json_schema = self._build_json_schema()

        self._log_action(
            state,
            "calling_llm_for_painpoints",
            details={
                "evidence_count": len(evidence[:self.MAX_EVIDENCE_ITEMS]),
                "model": self.config.openai_model
            }
        )

        try:
            # Call LLM with structured output
            llm_response = self._call_llm(
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                },
                temperature=0.4  # Moderate temperature for consistent extraction
            )

            # Parse response
            content = llm_response.get("content", "")
            if not content:
                self.logger.error("Empty response from LLM")
                return []

            # Parse JSON and validate with Pydantic
            data_dict = json.loads(content)
            result = PainPointsResponse(**data_dict)

            self.logger.info(
                f"Extracted {len(result.pain_points)} pain point clusters"
            )
            self._log_action(
                state,
                "llm_extraction_complete",
                details={
                    "clusters_found": len(result.pain_points),
                    "rationale": result.rationale[:200] if result.rationale else ""
                }
            )

            return result.pain_points

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to extract pain points: {e}")
            raise

    def _build_json_schema(self) -> Dict[str, Any]:
        """Build JSON schema for OpenAI structured output.

        Returns:
            JSON schema dictionary compatible with OpenAI API
        """
        return {
            "name": "pain_points_response",
            "description": "Clustered pain points extracted from research evidence",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "pain_points": {
                        "type": "array",
                        "description": "List of clustered pain points",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cluster_name": {
                                    "type": "string",
                                    "description": "Short, specific name for the cluster"
                                },
                                "who": {
                                    "type": "string",
                                    "description": "Specific user segment affected"
                                },
                                "what": {
                                    "type": "string",
                                    "description": "Concrete description of the pain point"
                                },
                                "why": {
                                    "type": "string",
                                    "description": "Root cause explanation"
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Severity level"
                                },
                                "frequency": {
                                    "type": "string",
                                    "description": "How often mentioned"
                                },
                                "example_quotes": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Direct quotes from evidence"
                                }
                            },
                            "required": [
                                "cluster_name", "who", "what", "why",
                                "severity", "frequency", "example_quotes"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief explanation of clustering approach"
                    }
                },
                "required": ["pain_points", "rationale"],
                "additionalProperties": False
            }
        }

    def _prepare_evidence_context(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence for LLM context.

        Args:
            evidence: List of evidence dictionaries

        Returns:
            Formatted string of evidence for prompt
        """
        context_parts = []

        for e in evidence:
            e_id = e.get('id', 'unknown')
            e_type = e.get('type', 'unknown')
            credibility = e.get('credibility', 'unknown')
            title = e.get('title', 'Untitled')[:200]
            snippet = e.get('snippet', '')[:800]
            full_text = e.get('full_text', '')[:1500]

            # Use full_text if available and longer
            content = full_text if len(full_text) > len(snippet) else snippet

            context_parts.append(
                f"Source {e_id} ({e_type}, credibility: {credibility}):\n"
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"---"
            )

        return "\n".join(context_parts)

    def _link_evidence(
        self,
        pain_point: ExtractedPainPoint,
        evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Link pain point back to evidence IDs based on quote matching.

        Args:
            pain_point: The extracted pain point with quotes
            evidence: List of evidence to search

        Returns:
            List of evidence IDs that support this pain point
        """
        linked_ids = set()

        for quote in pain_point.example_quotes:
            quote_lower = quote.lower().strip()

            # Skip very short quotes (likely not meaningful)
            if len(quote_lower) < 10:
                continue

            for e in evidence:
                e_id = e.get('id', '')
                snippet = e.get('snippet', '').lower()
                full_text = e.get('full_text', '').lower()

                # Check for substring match (handles minor variations)
                # Use first 50 chars of quote to handle truncation
                quote_prefix = quote_lower[:50]

                if quote_prefix in snippet or quote_prefix in full_text:
                    linked_ids.add(e_id)
                    continue

                # Check for word overlap (handles paraphrasing)
                quote_words = set(quote_lower.split())
                snippet_words = set(snippet.split())
                overlap = len(quote_words & snippet_words) / max(len(quote_words), 1)

                if overlap > 0.6:  # 60% word overlap threshold
                    linked_ids.add(e_id)

        # If no matches found, link to most relevant sources by type
        if not linked_ids:
            for e in evidence[:3]:
                e_id = e.get('id', '')
                if e_id:
                    linked_ids.add(e_id)

        return list(linked_ids)

    def _mark_task_done(self, state: State, task_id: str) -> None:
        """Mark a task as done on the task board.

        Args:
            state: Current state
            task_id: ID of task to mark done
        """
        for t in state.task_board:
            if t.id == task_id:
                t.status = "done"
                break

    def _mark_task_blocked(self, state: State, task_id: str) -> None:
        """Mark a task as blocked on the task board.

        Args:
            state: Current state
            task_id: ID of task to mark blocked
        """
        for t in state.task_board:
            if t.id == task_id:
                t.status = "blocked"
                break
