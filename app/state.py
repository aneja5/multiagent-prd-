"""State management for the multi-agent PRD generator.

This module defines the complete state schema using Pydantic models and provides
functions for state persistence and retrieval.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Query(BaseModel):
    """Represents a research query to be executed."""

    id: str
    text: str
    category: Literal["competitor", "pain_points", "workflow", "compliance"]
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["pending", "done"] = "pending"
    expected_sources: List[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Container for all research queries."""

    queries: List[Query] = Field(default_factory=list)


class Evidence(BaseModel):
    """Represents a piece of research evidence with source information."""

    id: str
    url: str
    title: str
    type: Literal["article", "forum", "docs", "pricing", "review"]
    snippet: str
    full_text: str = ""
    tags: List[str] = Field(default_factory=list)
    credibility: Literal["high", "medium", "low"] = "medium"
    query_id: str
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)


class PainPoint(BaseModel):
    """A user pain point identified from research.

    Supports both simple and detailed pain point representations.
    The detailed fields (cluster_name, who, what, why, example_quotes)
    are used by the PainPointsAgent for richer analysis.
    """

    id: Optional[str] = None
    description: str
    severity: Literal["critical", "major", "minor"]
    evidence_ids: List[str] = Field(default_factory=list)
    frequency: Optional[str] = None

    # Extended fields for detailed pain point analysis
    cluster_name: Optional[str] = None
    who: Optional[str] = None  # Which user segment
    what: Optional[str] = None  # The pain point details
    why: Optional[str] = None  # Root cause
    example_quotes: List[str] = Field(default_factory=list)


class Competitor(BaseModel):
    """Information about a competitor product.

    Supports both simple and detailed competitor representations.
    The detailed fields are used by the CompetitorsAgent for richer analysis.
    """

    id: Optional[str] = None
    name: str
    description: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    pricing: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)

    # Extended fields for detailed competitive analysis
    url: Optional[str] = None
    positioning: Optional[str] = None  # How they position themselves
    icp: Optional[str] = None  # Ideal customer profile
    pricing_model: Optional[str] = None  # Pricing structure
    pricing_details: Optional[str] = None  # Specific pricing info
    key_features: List[str] = Field(default_factory=list)


class Workflow(BaseModel):
    """A typical user workflow in the domain."""

    name: str
    steps: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class Insights(BaseModel):
    """Synthesized insights from research."""

    pain_points: List[PainPoint] = Field(default_factory=list)
    competitors: List[Competitor] = Field(default_factory=list)
    workflows: List[Workflow] = Field(default_factory=list)

    # Competitive analysis insights
    opportunity_gaps: List[str] = Field(default_factory=list)
    market_insights: str = ""


class PRD(BaseModel):
    """The final Product Requirements Document."""

    sections: Dict[str, Any] = Field(default_factory=dict)
    notion_markdown: str = ""
    citation_map: Dict[str, List[str]] = Field(default_factory=dict)


class Metadata(BaseModel):
    """Metadata about the product idea and generation preferences."""

    raw_idea: str
    domain: str = ""
    industry_tags: List[str] = Field(default_factory=list)
    target_user: str = ""
    geography: str = ""
    compliance_contexts: List[str] = Field(default_factory=list)
    prd_style: Literal["lean_mvp", "full"] = "lean_mvp"
    clarification_status: Literal["pending", "confirmed"] = "pending"


class Task(BaseModel):
    """A task in the agent task board."""

    id: str
    owner: str
    status: Literal["pending", "doing", "done", "blocked"] = "pending"
    description: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


class AgentTraceEntry(BaseModel):
    """A single entry in the agent execution trace."""

    agent: str
    turn: int
    action: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Optional[Dict[str, Any]] = None


class State(BaseModel):
    """The complete state of the PRD generation process.

    This is the shared state object that all agents read from and write to.
    """

    run_id: str
    created_at: str
    status: Literal["running", "blocked", "done"] = "running"

    metadata: Metadata
    research_plan: ResearchPlan = Field(default_factory=ResearchPlan)
    evidence: List[Evidence] = Field(default_factory=list)
    insights: Insights = Field(default_factory=Insights)
    prd: PRD = Field(default_factory=PRD)
    task_board: List[Task] = Field(default_factory=list)
    agent_trace: List[AgentTraceEntry] = Field(default_factory=list)

    def get_queries_by_category(self, category: str) -> List[Query]:
        """Get all queries in a specific category.

        Args:
            category: The category to filter by (competitor, pain_points, workflow, compliance)

        Returns:
            List of Query objects in the specified category
        """
        return [q for q in self.research_plan.queries if q.category == category]

    def get_high_priority_queries(self) -> List[Query]:
        """Get all high priority queries.

        Returns:
            List of Query objects with priority "high"
        """
        return [q for q in self.research_plan.queries if q.priority == "high"]

    def mark_query_done(self, query_id: str) -> None:
        """Mark a query as done by its ID.

        Args:
            query_id: The ID of the query to mark as done
        """
        for query in self.research_plan.queries:
            if query.id == query_id:
                query.status = "done"
                break

    def get_pending_queries(self) -> List[Query]:
        """Get all pending queries.

        Returns:
            List of Query objects with status "pending"
        """
        return [q for q in self.research_plan.queries if q.status == "pending"]

    def get_queries_by_priority(self, priority: str) -> List[Query]:
        """Get all queries with a specific priority.

        Args:
            priority: The priority to filter by (high, medium, low)

        Returns:
            List of Query objects with the specified priority
        """
        return [q for q in self.research_plan.queries if q.priority == priority]


def get_runs_dir() -> Path:
    """Get the directory for storing run data.

    Returns:
        Path to the runs directory
    """
    runs_dir = Path("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_state_file_path(run_id: str) -> Path:
    """Get the file path for a specific run's state.

    Args:
        run_id: The unique identifier for the run

    Returns:
        Path to the state file
    """
    return get_runs_dir() / f"{run_id}.json"


def save_state(state: State, run_id: Optional[str] = None) -> None:
    """Save the state to disk.

    Args:
        state: The state object to save
        run_id: Optional run ID (uses state.run_id if not provided)

    Raises:
        IOError: If the state cannot be saved
    """
    try:
        if run_id is None:
            run_id = state.run_id

        file_path = get_state_file_path(run_id)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state.model_dump(), f, indent=2, ensure_ascii=False)

    except Exception as e:
        raise IOError(f"Failed to save state for run {run_id}: {e}")


def load_state(run_id: str) -> State:
    """Load a state from disk.

    Args:
        run_id: The unique identifier for the run

    Returns:
        The loaded state object

    Raises:
        FileNotFoundError: If the state file doesn't exist
        ValueError: If the state file is invalid
    """
    try:
        file_path = get_state_file_path(run_id)

        if not file_path.exists():
            raise FileNotFoundError(f"No state found for run {run_id}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return State(**data)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid state file for run {run_id}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load state for run {run_id}: {e}")


def create_new_state(raw_idea: str) -> State:
    """Create a new state object for a fresh run.

    Args:
        raw_idea: The user's product idea description

    Returns:
        A new initialized state object
    """
    run_id = str(uuid.uuid4())

    return State(
        run_id=run_id,
        created_at=datetime.utcnow().isoformat(),
        status="running",
        metadata=Metadata(raw_idea=raw_idea)
    )


def list_all_runs() -> List[Dict[str, Any]]:
    """List all available runs with their metadata.

    Returns:
        List of run metadata dictionaries
    """
    runs_dir = get_runs_dir()
    runs = []

    for state_file in runs_dir.glob("*.json"):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            runs.append({
                "run_id": data.get("run_id"),
                "created_at": data.get("created_at"),
                "status": data.get("status"),
                "raw_idea": data.get("metadata", {}).get("raw_idea", "")[:100]
            })
        except Exception:
            # Skip invalid files
            continue

    # Sort by creation time, newest first
    runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return runs
