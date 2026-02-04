"""CompetitorsAgent - analyzes competitive landscape.

This agent analyzes competitor evidence (pricing pages, reviews, comparison articles)
to create a structured competitive analysis with positioning, features, and gaps.
"""

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from rich.console import Console

from agents.base_agent import BaseAgent
from app.logger import get_logger
from app.state import Competitor, State, Task

logger = get_logger(__name__)
console = Console()


class ExtractedCompetitor(BaseModel):
    """Structured competitor analysis extracted by LLM."""

    name: str = Field(description="Official company/product name")
    url: Optional[str] = Field(default=None, description="Official website URL")
    positioning: str = Field(description="How they position themselves (1-2 sentences)")
    icp: str = Field(description="Ideal customer profile (be specific)")
    pricing_model: str = Field(
        description="Pricing structure (e.g., 'Freemium + paid tiers', 'Per-user per-month')"
    )
    pricing_details: Optional[str] = Field(
        default=None,
        description="Specific pricing if available"
    )
    key_features: List[str] = Field(
        description="Top 5-7 features (be specific)",
        min_length=3,
        max_length=10
    )
    strengths: List[str] = Field(
        description="What they do well (2-4 items)",
        min_length=1,
        max_length=5
    )
    weaknesses: List[str] = Field(
        description="What they lack or do poorly (2-4 items)",
        min_length=1,
        max_length=5
    )


class CompetitiveAnalysis(BaseModel):
    """LLM response format for competitive analysis."""

    competitors: List[ExtractedCompetitor] = Field(
        description="List of analyzed competitors"
    )
    opportunity_gaps: List[str] = Field(
        description="What competitors collectively miss (3-6 items)",
        min_length=1,
        max_length=10
    )
    market_insights: str = Field(
        description="Overall market observations (2-3 sentences)"
    )


class CompetitorsAgent(BaseAgent):
    """Analyze competitive landscape from evidence.

    Process:
    1. Filter evidence for competitor sources (pricing, reviews, comparisons)
    2. Extract competitor information using LLM
    3. Identify positioning, features, pricing
    4. Analyze collective gaps (opportunities)
    5. Update state.insights.competitors

    Attributes:
        name: Agent identifier ("competitors")
        client: OpenAI client for API calls
    """

    # Evidence types relevant for competitor analysis
    RELEVANT_EVIDENCE_TYPES = {"pricing", "review", "article", "docs"}

    # Tags that indicate competitor relevance
    RELEVANT_TAGS = {"competitor", "comparison", "alternative", "pricing"}

    # Maximum evidence items to process
    MAX_EVIDENCE_ITEMS = 30

    def __init__(self, name: str, client: OpenAI) -> None:
        """Initialize the CompetitorsAgent.

        Args:
            name: Agent identifier (typically "competitors")
            client: Configured OpenAI client instance
        """
        super().__init__(name, client)
        self.logger = get_logger(__name__)

    def run(self, state: State) -> State:
        """Execute competitive analysis.

        Args:
            state: Current shared state containing evidence

        Returns:
            Updated state with competitors added to insights

        Raises:
            Exception: If analysis fails after retries
        """
        self.logger.info("Starting competitive analysis")
        self._log_action(state, "started_competitor_analysis")

        # Create task on task board
        task = Task(
            id=f"T-COMPETITORS-{state.run_id[:8]}",
            owner="competitors",
            status="doing",
            description="Analyze competitive landscape from evidence"
        )
        state.task_board.append(task)

        try:
            # 1. Think: Filter relevant evidence
            relevant_evidence = self._filter_evidence(state.evidence)

            if not relevant_evidence:
                self.logger.warning("No relevant evidence for competitor analysis")
                console.print("[yellow]No competitor evidence found[/yellow]")
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
                f"for competitors...[/bold cyan]\n"
            )

            # 2. Act: Analyze competitors using LLM
            analysis = self._analyze_competitors(relevant_evidence, state)

            if not analysis:
                self.logger.warning("No competitive analysis returned")
                console.print("[yellow]Could not analyze competitors[/yellow]")
                self._mark_task_done(state, task.id)
                return state

            # 3. Observe & Update: Add competitors to state
            for i, comp in enumerate(analysis.competitors):
                # Link competitor to evidence
                evidence_ids = self._link_evidence(comp.name, relevant_evidence)

                # Create Competitor for state
                competitor = Competitor(
                    id=f"C{len(state.insights.competitors) + 1}",
                    name=comp.name,
                    url=comp.url,
                    positioning=comp.positioning,
                    icp=comp.icp,
                    pricing_model=comp.pricing_model,
                    pricing_details=comp.pricing_details,
                    pricing=comp.pricing_details,  # Also set legacy field
                    description=comp.positioning,  # Use positioning as description
                    key_features=comp.key_features,
                    strengths=comp.strengths,
                    weaknesses=comp.weaknesses,
                    evidence_ids=evidence_ids
                )

                state.insights.competitors.append(competitor)

                self.logger.debug(
                    f"Added competitor: {comp.name} "
                    f"(linked to {len(evidence_ids)} sources)"
                )

            # Add opportunity gaps and market insights
            state.insights.opportunity_gaps = analysis.opportunity_gaps
            state.insights.market_insights = analysis.market_insights

            # 4. Reflect: Log completion
            console.print(
                f"[green]Analyzed {len(analysis.competitors)} competitors[/green]"
            )
            console.print(
                f"[green]Identified {len(analysis.opportunity_gaps)} opportunity gaps[/green]\n"
            )

            self._log_action(
                state,
                "completed_competitor_analysis",
                details={
                    "competitors_found": len(analysis.competitors),
                    "opportunity_gaps": len(analysis.opportunity_gaps),
                    "competitor_names": [c.name for c in analysis.competitors]
                }
            )

            # Mark task as done
            self._mark_task_done(state, task.id)

            self.logger.info(
                f"Competitive analysis complete: {len(analysis.competitors)} competitors"
            )

            return state

        except Exception as e:
            # Mark task as blocked on error
            self._mark_task_blocked(state, task.id)
            self.logger.error(f"Competitive analysis failed: {e}")
            self._log_action(state, f"competitor_analysis_failed: {str(e)}")
            raise

    def _filter_evidence(self, evidence: List[Any]) -> List[Dict[str, Any]]:
        """Filter evidence for competitor relevant sources.

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

    def _analyze_competitors(
        self,
        evidence: List[Dict[str, Any]],
        state: State
    ) -> Optional[CompetitiveAnalysis]:
        """Use LLM to analyze competitors.

        Args:
            evidence: Filtered relevant evidence
            state: Current state for context

        Returns:
            CompetitiveAnalysis object or None on failure
        """
        # Load prompt template
        try:
            prompt_template = self._load_prompt()
        except FileNotFoundError:
            self.logger.error("Competitors prompt file not found")
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
                    "You are a competitive analyst researching market positioning. "
                    "Analyze the evidence carefully and provide detailed competitor analysis."
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
            "calling_llm_for_competitors",
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
                return None

            # Parse JSON and validate with Pydantic
            data_dict = json.loads(content)
            result = CompetitiveAnalysis(**data_dict)

            self.logger.info(
                f"Analyzed {len(result.competitors)} competitors, "
                f"found {len(result.opportunity_gaps)} gaps"
            )
            self._log_action(
                state,
                "llm_analysis_complete",
                details={
                    "competitors_found": len(result.competitors),
                    "gaps_found": len(result.opportunity_gaps)
                }
            )

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to analyze competitors: {e}")
            raise

    def _build_json_schema(self) -> Dict[str, Any]:
        """Build JSON schema for OpenAI structured output.

        Returns:
            JSON schema dictionary compatible with OpenAI API
        """
        return {
            "name": "competitive_analysis",
            "description": "Competitive landscape analysis from research evidence",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "competitors": {
                        "type": "array",
                        "description": "List of analyzed competitors",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Official company/product name"
                                },
                                "url": {
                                    "type": ["string", "null"],
                                    "description": "Official website URL"
                                },
                                "positioning": {
                                    "type": "string",
                                    "description": "How they position themselves"
                                },
                                "icp": {
                                    "type": "string",
                                    "description": "Ideal customer profile"
                                },
                                "pricing_model": {
                                    "type": "string",
                                    "description": "Pricing structure"
                                },
                                "pricing_details": {
                                    "type": ["string", "null"],
                                    "description": "Specific pricing if available"
                                },
                                "key_features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Top 5-7 features"
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "What they do well"
                                },
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "What they lack or do poorly"
                                }
                            },
                            "required": [
                                "name", "url", "positioning", "icp",
                                "pricing_model", "pricing_details",
                                "key_features", "strengths", "weaknesses"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "opportunity_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What competitors collectively miss"
                    },
                    "market_insights": {
                        "type": "string",
                        "description": "Overall market observations"
                    }
                },
                "required": ["competitors", "opportunity_gaps", "market_insights"],
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
            url = e.get('url', '')[:200]
            title = e.get('title', 'Untitled')[:200]
            snippet = e.get('snippet', '')[:800]
            full_text = e.get('full_text', '')[:1500]

            # Use full_text if available and longer
            content = full_text if len(full_text) > len(snippet) else snippet

            context_parts.append(
                f"Source {e_id} ({e_type}, credibility: {credibility}):\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
                f"---"
            )

        return "\n".join(context_parts)

    def _link_evidence(
        self,
        competitor_name: str,
        evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Link competitor to evidence where mentioned.

        Args:
            competitor_name: Name of the competitor
            evidence: List of evidence to search

        Returns:
            List of evidence IDs that mention this competitor
        """
        linked_ids = set()
        name_lower = competitor_name.lower()

        # Also check for common variations
        name_parts = name_lower.split()

        for e in evidence:
            e_id = e.get('id', '')
            title = e.get('title', '').lower()
            snippet = e.get('snippet', '').lower()
            full_text = e.get('full_text', '').lower()
            url = e.get('url', '').lower()

            # Check full name
            if name_lower in title or name_lower in snippet or name_lower in full_text:
                linked_ids.add(e_id)
                continue

            # Check if name appears in URL (e.g., freshbooks.com)
            if name_lower.replace(' ', '') in url:
                linked_ids.add(e_id)
                continue

            # Check first word of name (handles "FreshBooks" matching "freshbooks")
            if name_parts and name_parts[0] in title + snippet + full_text:
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
