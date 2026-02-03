"""Orchestrator for coordinating multiple agents using DAG-based task execution.

This module manages the execution flow of all agents based on task dependencies,
handles retries, checkpointing, and ensures agents work together to generate the PRD.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.base_agent import BaseAgent
from agents.clarification import ClarificationAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from app.config import get_config
from app.logger import get_logger
from app.state import AgentTraceEntry, State, Task, save_state


# Task dependency DAG - defines which tasks must complete before others can run
TASK_DEPENDENCIES: Dict[str, List[str]] = {
    "clarification": [],
    "planning": ["clarification"],
    "research": ["planning"],           # Coming in Day 3
    "painpoints": ["research"],         # Coming later
    "competitors": ["research"],        # Coming later
    "prd_draft": ["painpoints", "competitors"],  # Coming later
    "validation": ["prd_draft"],        # Coming later
}

# Agent registry - maps task owners to agent classes
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "clarification": ClarificationAgent,
    "planning": PlannerAgent,
    "research": ResearcherAgent,
    # More agents coming in future days:
    # "painpoints": PainPointsAgent,
    # "competitors": CompetitorAgent,
    # "prd_draft": PRDDraftAgent,
    # "validation": ValidationAgent,
}

# Task definitions for initial task board
INITIAL_TASKS = [
    {"id": "T1", "owner": "clarification", "description": "Extract product metadata"},
    {"id": "T2", "owner": "planning", "description": "Generate research queries"},
    {"id": "T3", "owner": "research", "description": "Execute queries and collect evidence"},
    {"id": "T4", "owner": "painpoints", "description": "Extract and cluster pain points"},
    {"id": "T5", "owner": "competitors", "description": "Analyze competitive landscape"},
    {"id": "T6", "owner": "prd_draft", "description": "Write PRD with citations"},
    {"id": "T7", "owner": "validation", "description": "Validate citations and claims"},
]


class OrchestratorError(Exception):
    """Raised when orchestration encounters a critical error."""
    pass


class Orchestrator:
    """Coordinates the execution of multiple agents using DAG-based task scheduling.

    The orchestrator manages:
    - Task dependency resolution
    - Agent instantiation and execution
    - Retry logic for failed agents
    - State checkpointing after each agent
    - Progress tracking with rich output

    Attributes:
        client: OpenAI client for agent use
        agents: Dictionary of instantiated agents by name
        config: Application configuration
        logger: Logger instance
        console: Rich console for output
        max_retries: Maximum retry attempts per agent
        retry_counts: Track retry attempts per task
    """

    MAX_RETRIES = 3

    def __init__(self, client: OpenAI, console: Optional[Console] = None) -> None:
        """Initialize the orchestrator.

        Args:
            client: Configured OpenAI client instance
            console: Optional Rich console for output (creates one if not provided)
        """
        self.client = client
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.console = console or Console()
        self.retry_counts: Dict[str, int] = {}

        # Instantiate available agents
        self.agents: Dict[str, BaseAgent] = {}
        for name, agent_class in AGENT_REGISTRY.items():
            self.agents[name] = agent_class(name, client)
            self.logger.debug(f"Instantiated agent: {name}")

    def run(self, state: State, max_iterations: int = 20) -> State:
        """Execute the orchestration loop.

        Main orchestration loop:
        1. Check task_board for completed tasks
        2. Find next runnable task (all dependencies done)
        3. Execute corresponding agent
        4. Update task status
        5. Save checkpoint
        6. Repeat until all done or max_iterations

        Args:
            state: The current state
            max_iterations: Maximum number of iterations to prevent infinite loops

        Returns:
            The final state after orchestration
        """
        self.logger.info(f"Starting orchestration for run {state.run_id}")
        self.logger.info(f"Available agents: {list(self.agents.keys())}")

        # Initialize task board if empty
        if not state.task_board:
            self._initialize_task_board(state)
            save_state(state)

        iteration = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            progress_task = progress.add_task("[cyan]Orchestrating agents...", total=None)

            while iteration < max_iterations:
                iteration += 1
                self.logger.debug(f"Orchestration iteration {iteration}/{max_iterations}")

                # Check if we're done
                if self._is_complete(state):
                    self.logger.info("All tasks complete or blocked")
                    state.status = "done" if self._all_tasks_done(state) else "blocked"
                    break

                # Find next runnable task
                next_task_owner = self._get_next_task(state)

                if next_task_owner is None:
                    # No runnable tasks but not complete - might be waiting for unimplemented agents
                    self.logger.info("No runnable tasks available")

                    # Check if all implemented agents have finished
                    pending_implemented = self._get_pending_implemented_tasks(state)
                    if not pending_implemented:
                        self.logger.info("All implemented agents have completed")
                        state.status = "done"
                    else:
                        self.logger.warning(f"Blocked: waiting for tasks {pending_implemented}")
                        state.status = "blocked"
                    break

                # Update progress display
                progress.update(
                    progress_task,
                    description=f"[cyan]Running {next_task_owner} agent..."
                )

                # Execute the agent
                self.logger.info(f"Executing agent: {next_task_owner}")
                state = self._execute_agent(next_task_owner, state)

                # Save checkpoint
                save_state(state)

            if iteration >= max_iterations:
                self.logger.warning(
                    f"Orchestration stopped: reached max iterations ({max_iterations})"
                )
                state.status = "blocked"
                self._log_orchestrator_action(
                    state,
                    f"Stopped: max iterations ({max_iterations}) reached"
                )

        self.logger.info(f"Orchestration completed with status: {state.status}")
        return state

    def _initialize_task_board(self, state: State) -> None:
        """Initialize the task board with all tasks.

        Args:
            state: The state to initialize
        """
        self.logger.info("Initializing task board")

        for task_def in INITIAL_TASKS:
            task = Task(
                id=task_def["id"],
                owner=task_def["owner"],
                status="pending",
                description=task_def["description"]
            )
            state.task_board.append(task)

        self._log_orchestrator_action(
            state,
            f"Initialized task board with {len(INITIAL_TASKS)} tasks"
        )

    def _get_next_task(self, state: State) -> Optional[str]:
        """Find the next task where all dependencies are satisfied.

        A task is runnable if:
        - Its status is "pending"
        - All tasks it depends on have status "done"
        - An agent exists to handle it

        Args:
            state: The current state

        Returns:
            Task owner name or None if no runnable task
        """
        for task in state.task_board:
            if task.status != "pending":
                continue

            # Check if agent exists for this task
            if task.owner not in self.agents:
                self.logger.debug(f"No agent for task {task.id} (owner: {task.owner})")
                continue

            # Check dependencies
            dependencies = TASK_DEPENDENCIES.get(task.owner, [])
            deps_satisfied = True

            for dep in dependencies:
                dep_task = self._find_task_by_owner(state, dep)
                if dep_task is None or dep_task.status != "done":
                    deps_satisfied = False
                    self.logger.debug(
                        f"Task {task.id} blocked: dependency {dep} not done"
                    )
                    break

            if deps_satisfied:
                return task.owner

        return None

    def _find_task_by_owner(self, state: State, owner: str) -> Optional[Task]:
        """Find a task by its owner.

        Args:
            state: The current state
            owner: The task owner to find

        Returns:
            Task object or None
        """
        for task in state.task_board:
            if task.owner == owner:
                return task
        return None

    def _execute_agent(self, agent_name: str, state: State) -> State:
        """Run an agent and handle errors with retry logic.

        - Try to execute agent
        - If success: mark task as done
        - If failure: increment retry count, mark blocked after MAX_RETRIES failures

        Args:
            agent_name: Name of the agent to execute
            state: The current state

        Returns:
            Updated state
        """
        agent = self.agents.get(agent_name)
        if agent is None:
            self.logger.error(f"No agent found for: {agent_name}")
            return state

        task = self._find_task_by_owner(state, agent_name)
        if task is None:
            self.logger.error(f"No task found for agent: {agent_name}")
            return state

        # Mark task as in progress
        task.status = "doing"
        self._log_orchestrator_action(state, f"Starting agent: {agent_name}")

        try:
            # Execute the agent
            state = agent.run(state)

            # Mark task as done
            task.status = "done"
            task.completed_at = datetime.utcnow().isoformat()

            # Reset retry count on success
            self.retry_counts[agent_name] = 0

            self._log_orchestrator_action(
                state,
                f"Agent {agent_name} completed successfully"
            )

            self.console.print(f"  [green]✓[/green] {agent_name} completed")

        except Exception as e:
            self.logger.error(f"Agent {agent_name} failed: {e}")

            # Increment retry count
            self.retry_counts[agent_name] = self.retry_counts.get(agent_name, 0) + 1
            retries = self.retry_counts[agent_name]

            if retries >= self.MAX_RETRIES:
                # Max retries exceeded - mark as blocked
                task.status = "blocked"
                self._log_orchestrator_action(
                    state,
                    f"Agent {agent_name} blocked after {retries} failures: {str(e)}"
                )
                self.console.print(
                    f"  [red]✗[/red] {agent_name} blocked after {retries} retries"
                )
            else:
                # Reset to pending for retry
                task.status = "pending"
                self._log_orchestrator_action(
                    state,
                    f"Agent {agent_name} failed (attempt {retries}/{self.MAX_RETRIES}): {str(e)}"
                )
                self.console.print(
                    f"  [yellow]![/yellow] {agent_name} failed, will retry "
                    f"({retries}/{self.MAX_RETRIES})"
                )

        return state

    def _is_complete(self, state: State) -> bool:
        """Check if orchestration should stop.

        Orchestration is complete when:
        - All tasks are done, OR
        - All tasks are either done or blocked, OR
        - No more runnable tasks exist

        Args:
            state: The current state

        Returns:
            True if orchestration should stop
        """
        for task in state.task_board:
            if task.status in ["pending", "doing"]:
                # Check if this task could potentially run
                if task.owner in self.agents:
                    deps = TASK_DEPENDENCIES.get(task.owner, [])
                    deps_blocked = any(
                        self._find_task_by_owner(state, d).status == "blocked"
                        for d in deps
                        if self._find_task_by_owner(state, d) is not None
                    )
                    if not deps_blocked:
                        return False
        return True

    def _all_tasks_done(self, state: State) -> bool:
        """Check if all implemented tasks are done.

        Args:
            state: The current state

        Returns:
            True if all tasks with available agents are done
        """
        for task in state.task_board:
            if task.owner in self.agents and task.status != "done":
                return False
        return True

    def _get_pending_implemented_tasks(self, state: State) -> List[str]:
        """Get list of pending tasks that have implemented agents.

        Args:
            state: The current state

        Returns:
            List of task owners that are pending and have agents
        """
        pending = []
        for task in state.task_board:
            if task.status == "pending" and task.owner in self.agents:
                pending.append(task.owner)
        return pending

    def _log_orchestrator_action(
        self,
        state: State,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an entry to the agent trace for orchestrator actions.

        Args:
            state: The current state
            action: Description of the action
            details: Optional additional details
        """
        trace_entry = AgentTraceEntry(
            agent="orchestrator",
            turn=len([e for e in state.agent_trace if e.agent == "orchestrator"]) + 1,
            action=action,
            details=details
        )
        state.agent_trace.append(trace_entry)
        self.logger.info(f"[Orchestrator] {action}")

    def get_task_status_summary(self, state: State) -> Dict[str, int]:
        """Get a summary of task statuses.

        Args:
            state: The current state

        Returns:
            Dictionary with counts by status
        """
        summary = {"pending": 0, "doing": 0, "done": 0, "blocked": 0}
        for task in state.task_board:
            summary[task.status] = summary.get(task.status, 0) + 1
        return summary


def initialize_task_board(state: State) -> None:
    """Initialize the task board with all tasks.

    This is a standalone function that can be called from create_new_state.

    Args:
        state: The state to initialize
    """
    for task_def in INITIAL_TASKS:
        task = Task(
            id=task_def["id"],
            owner=task_def["owner"],
            status="pending",
            description=task_def["description"]
        )
        state.task_board.append(task)
