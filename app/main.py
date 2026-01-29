"""Main CLI interface for the multi-agent PRD generator.

This module provides the command-line interface for creating new PRD generation
runs, resuming existing runs, and listing all runs.
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from agents.clarification import ClarificationAgent
from app.config import ConfigurationError, get_config
from app.logger import get_logger, log_error, log_info, log_section, log_success
from app.orchestrator import Orchestrator
from app.state import (
    State,
    create_new_state,
    list_all_runs,
    load_state,
    save_state,
)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Multi-agent PRD Generator - Generate research-backed Product Requirements Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new PRD generation
  python -m app.main "Build a project management tool for remote teams"

  # Resume an existing run
  python -m app.main --resume abc123-def456-...

  # List all runs
  python -m app.main --list
        """
    )

    parser.add_argument(
        "idea",
        nargs="?",
        help="Product idea description (required for new runs)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        metavar="RUN_ID",
        help="Resume an existing run by ID"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available runs"
    )

    return parser


def display_runs_table(console: Console) -> None:
    """Display a table of all available runs.

    Args:
        console: Rich console instance for output
    """
    runs = list_all_runs()

    if not runs:
        log_info("No runs found. Start a new run by providing a product idea.")
        return

    table = Table(title="Available Runs", show_header=True, header_style="bold cyan")
    table.add_column("Run ID", style="dim")
    table.add_column("Created At", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Idea", style="white", no_wrap=False)

    for run in runs:
        run_id = run.get("run_id", "")[:8] + "..."  # Truncate for display
        created_at = run.get("created_at", "")[:19]  # Remove microseconds
        status = run.get("status", "unknown")
        idea = run.get("raw_idea", "")[:80] + ("..." if len(run.get("raw_idea", "")) > 80 else "")

        # Color code status
        status_style = {
            "running": "[yellow]running[/yellow]",
            "blocked": "[red]blocked[/red]",
            "done": "[green]done[/green]"
        }.get(status, status)

        table.add_row(run.get("run_id", ""), created_at, status_style, idea)

    console.print()
    console.print(table)
    console.print()
    log_info(f"Total runs: {len(runs)}")


def display_run_summary(state: State, console: Console) -> None:
    """Display a summary of the current run.

    Args:
        state: The state to summarize
        console: Rich console instance for output
    """
    log_section("Run Summary", console)

    console.print(f"[bold]Run ID:[/bold] {state.run_id}")
    console.print(f"[bold]Status:[/bold] {state.status}")
    console.print(f"[bold]Created:[/bold] {state.created_at[:19]}")
    console.print(f"\n[bold]Product Idea:[/bold]\n{state.metadata.raw_idea}")

    # Show task board if available
    if state.task_board:
        console.print(f"\n[bold]Tasks:[/bold] {len(state.task_board)} total")
        pending = sum(1 for t in state.task_board if t.status == "pending")
        doing = sum(1 for t in state.task_board if t.status == "doing")
        done = sum(1 for t in state.task_board if t.status == "done")
        blocked = sum(1 for t in state.task_board if t.status == "blocked")

        console.print(f"  - Pending: {pending}")
        console.print(f"  - In Progress: {doing}")
        console.print(f"  - Done: {done}")
        console.print(f"  - Blocked: {blocked}")

    # Show research progress
    if state.research_plan.queries:
        console.print(f"\n[bold]Research Queries:[/bold] {len(state.research_plan.queries)}")

    if state.evidence:
        console.print(f"[bold]Evidence Collected:[/bold] {len(state.evidence)}")

    # Show agent activity
    if state.agent_trace:
        console.print(f"\n[bold]Agent Actions:[/bold] {len(state.agent_trace)}")
        last_action = state.agent_trace[-1]
        console.print(f"  Last: {last_action.agent} - {last_action.action}")

    console.print()


def create_new_run(idea: str, console: Console) -> int:
    """Create and execute a new PRD generation run.

    Args:
        idea: The product idea description
        console: Rich console instance for output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        log_section("Starting New PRD Generation", console)
        log_info(f"Product Idea: {idea}")

        # Create new state
        state = create_new_state(idea)
        log_success(f"Created new run: {state.run_id}")

        # Save initial state
        save_state(state)

        # Initialize OpenAI client
        config = get_config()
        client = OpenAI(api_key=config.openai_api_key)

        # Initialize orchestrator
        orchestrator = Orchestrator(client)

        # Register agents
        log_info("Registering agents...")
        orchestrator.register_agent(ClarificationAgent("clarification", client))
        # TODO: Register additional agents here
        # orchestrator.register_agent(ResearchPlannerAgent("research_planner", client))
        # ... etc

        log_info("Starting orchestration...")

        # Run orchestration
        final_state = orchestrator.run(state)

        # Save final state
        save_state(final_state)

        # Display summary
        display_run_summary(final_state, console)

        if final_state.status == "done":
            log_success("PRD generation completed successfully!")
            log_info(f"Output saved to: data/runs/{final_state.run_id}.json")
            return 0
        else:
            log_error(f"PRD generation ended with status: {final_state.status}")
            return 1

    except Exception as e:
        log_error(f"Error during PRD generation: {e}")
        return 1


def resume_run(run_id: str, console: Console) -> int:
    """Resume an existing PRD generation run.

    Args:
        run_id: The run ID to resume
        console: Rich console instance for output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        log_section("Resuming PRD Generation", console)
        log_info(f"Run ID: {run_id}")

        # Load existing state
        state = load_state(run_id)
        log_success("Loaded existing run state")

        # Display current status
        display_run_summary(state, console)

        if state.status == "done":
            log_info("This run is already complete.")
            return 0

        # Initialize OpenAI client
        config = get_config()
        client = OpenAI(api_key=config.openai_api_key)

        # Initialize orchestrator
        orchestrator = Orchestrator(client)

        # Register agents
        log_info("Registering agents...")
        orchestrator.register_agent(ClarificationAgent("clarification", client))
        # TODO: Register additional agents here

        log_info("Resuming orchestration...")

        # Run orchestration
        final_state = orchestrator.run(state)

        # Save final state
        save_state(final_state)

        # Display summary
        display_run_summary(final_state, console)

        if final_state.status == "done":
            log_success("PRD generation completed successfully!")
            return 0
        else:
            log_error(f"PRD generation ended with status: {final_state.status}")
            return 1

    except FileNotFoundError as e:
        log_error(str(e))
        return 1
    except Exception as e:
        log_error(f"Error resuming run: {e}")
        return 1


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = setup_argument_parser()
    args = parser.parse_args()

    console = Console()

    try:
        # Initialize configuration
        config = get_config()
        logger = get_logger(__name__)

    except ConfigurationError as e:
        log_error(f"Configuration error: {e}")
        log_info("Please check your .env file or environment variables.")
        return 1

    # Handle --list
    if args.list:
        display_runs_table(console)
        return 0

    # Handle --resume
    if args.resume:
        return resume_run(args.resume, console)

    # Handle new run
    if args.idea:
        return create_new_run(args.idea, console)

    # No valid arguments provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
