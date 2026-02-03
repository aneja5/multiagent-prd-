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

  # Show detailed agent trace
  python -m app.main "HIPAA-compliant patient portal" --verbose

  # Resume an existing run
  python -m app.main --resume abc123-def456-...

  # List all runs
  python -m app.main --list

  # Inspect evidence from a run
  python -m app.main --inspect <run_id>
  python -m app.main --inspect <run_id> --type forum
  python -m app.main --inspect <run_id> --credibility high
  python -m app.main --inspect <run_id> --evidence-id E5
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

    parser.add_argument(
        "--inspect",
        type=str,
        metavar="RUN_ID",
        help="Inspect evidence collected from a run"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["article", "forum", "docs", "pricing", "review"],
        help="Filter evidence by type (use with --inspect)"
    )

    parser.add_argument(
        "--credibility",
        type=str,
        choices=["high", "medium", "low"],
        help="Filter evidence by credibility tier (use with --inspect)"
    )

    parser.add_argument(
        "--evidence-id",
        type=str,
        metavar="ID",
        help="Show full details for specific evidence (use with --inspect)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed agent trace and execution logs"
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


def display_run_summary(state: State, console: Console, verbose: bool = False) -> None:
    """Display a summary of the current run.

    Args:
        state: The state to summarize
        console: Rich console instance for output
        verbose: If True, show detailed agent trace
    """
    log_section("Run Summary", console)

    console.print(f"[bold]Run ID:[/bold] {state.run_id}")
    console.print(f"[bold]Status:[/bold] {state.status}")
    console.print(f"[bold]Created:[/bold] {state.created_at[:19]}")
    console.print(f"\n[bold]Product Idea:[/bold]\n{state.metadata.raw_idea}")

    # Show extracted metadata if available
    if state.metadata.domain:
        console.print("\n")
        log_success("Clarification Complete", console)

        table = Table(title="Extracted Metadata", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Domain", state.metadata.domain)
        table.add_row("Industry Tags", ", ".join(state.metadata.industry_tags))
        table.add_row("Target User", state.metadata.target_user)
        table.add_row("Geography", state.metadata.geography)

        compliance = ", ".join(state.metadata.compliance_contexts) if state.metadata.compliance_contexts else "None"
        table.add_row("Compliance", compliance)
        table.add_row("Status", state.metadata.clarification_status)

        console.print(table)

    # Show task board if available
    if state.task_board:
        console.print()
        task_table = Table(title="Task Board", show_header=True, header_style="bold cyan")
        task_table.add_column("ID", style="dim", width=6)
        task_table.add_column("Task", style="white", width=40)
        task_table.add_column("Owner", style="cyan", width=15)
        task_table.add_column("Status", style="yellow", width=10)

        for task in state.task_board:
            status_color = {
                "done": "green",
                "doing": "yellow",
                "pending": "dim",
                "blocked": "red"
            }.get(task.status, "white")

            task_table.add_row(
                task.id,
                task.description,
                task.owner,
                f"[{status_color}]{task.status}[/{status_color}]"
            )

        console.print(task_table)

        # Summary counts
        pending = sum(1 for t in state.task_board if t.status == "pending")
        doing = sum(1 for t in state.task_board if t.status == "doing")
        done = sum(1 for t in state.task_board if t.status == "done")
        blocked = sum(1 for t in state.task_board if t.status == "blocked")

        console.print(f"\n[dim]Summary: {done} done, {doing} in progress, {pending} pending, {blocked} blocked[/dim]")

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

        # Show detailed trace if verbose
        if verbose:
            console.print("\n[bold cyan]Agent Trace:[/bold cyan]")
            trace_table = Table(show_header=True, header_style="bold cyan")
            trace_table.add_column("Turn", style="dim", width=6)
            trace_table.add_column("Agent", style="yellow", width=15)
            trace_table.add_column("Action", style="white")

            for entry in state.agent_trace:
                trace_table.add_row(
                    str(entry.turn),
                    entry.agent,
                    entry.action[:80] + "..." if len(entry.action) > 80 else entry.action
                )

            console.print(trace_table)

    console.print()


def create_new_run(idea: str, console: Console, verbose: bool = False) -> int:
    """Create and execute a new PRD generation run.

    Args:
        idea: The product idea description
        console: Rich console instance for output
        verbose: If True, show detailed execution logs

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

        # Initialize orchestrator (agents are auto-registered)
        orchestrator = Orchestrator(client, console=console)

        log_info("Starting orchestration...")
        log_info(f"Available agents: {list(orchestrator.agents.keys())}")

        # Run orchestration
        final_state = orchestrator.run(state)

        # Save final state
        save_state(final_state)

        # Display summary
        display_run_summary(final_state, console, verbose=verbose)

        if final_state.status == "done":
            log_success("All available agents completed successfully!")
            log_info(f"Output saved to: data/runs/{final_state.run_id}.json")

            # Show what was accomplished
            if final_state.research_plan.queries:
                console.print(f"\n[green]Generated {len(final_state.research_plan.queries)} research queries[/green]")

            # Show next steps for unimplemented agents
            pending_tasks = [t for t in final_state.task_board if t.status == "pending"]
            if pending_tasks:
                console.print("\n[yellow]Pending tasks (agents not yet implemented):[/yellow]")
                for task in pending_tasks:
                    console.print(f"  [dim]- {task.description} ({task.owner})[/dim]")

            return 0
        else:
            log_error(f"PRD generation ended with status: {final_state.status}")
            return 1

    except Exception as e:
        log_error(f"Error during PRD generation: {e}")
        return 1


def resume_run(run_id: str, console: Console, verbose: bool = False) -> int:
    """Resume an existing PRD generation run.

    Args:
        run_id: The run ID to resume
        console: Rich console instance for output
        verbose: If True, show detailed execution logs

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
        display_run_summary(state, console, verbose=verbose)

        if state.status == "done":
            log_info("This run is already complete.")
            return 0

        # Initialize OpenAI client
        config = get_config()
        client = OpenAI(api_key=config.openai_api_key)

        # Initialize orchestrator (agents are auto-registered)
        orchestrator = Orchestrator(client, console=console)

        log_info("Resuming orchestration...")
        log_info(f"Available agents: {list(orchestrator.agents.keys())}")

        # Run orchestration
        final_state = orchestrator.run(state)

        # Save final state
        save_state(final_state)

        # Display summary
        display_run_summary(final_state, console, verbose=verbose)

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


def inspect_evidence(
    run_id: str,
    console: Console,
    type_filter: str = None,
    credibility_filter: str = None,
    evidence_id: str = None
) -> int:
    """Inspect evidence collected from a run.

    Args:
        run_id: The run ID to inspect
        console: Rich console instance for output
        type_filter: Filter by evidence type
        credibility_filter: Filter by credibility tier
        evidence_id: Show full details for specific evidence

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Load state
        state = load_state(run_id)

        if not state.evidence:
            log_info(f"No evidence collected for run {run_id[:8]}...")
            return 0

        # If specific evidence ID requested, show full details
        if evidence_id:
            return display_evidence_detail(state, evidence_id, console)

        # Apply filters
        evidence = list(state.evidence)

        if type_filter:
            evidence = [e for e in evidence if e.type == type_filter]

        if credibility_filter:
            evidence = [e for e in evidence if e.credibility == credibility_filter]

        # Display summary
        log_section(f"Evidence Inspection: {run_id[:8]}...", console)

        console.print(f"[bold]Product Idea:[/bold] {state.metadata.raw_idea[:80]}...")
        console.print(f"[bold]Domain:[/bold] {state.metadata.domain}")
        console.print()

        # Show filter status
        if type_filter or credibility_filter:
            filters = []
            if type_filter:
                filters.append(f"type={type_filter}")
            if credibility_filter:
                filters.append(f"credibility={credibility_filter}")
            console.print(f"[yellow]Filters applied:[/yellow] {', '.join(filters)}")
            console.print(f"[bold]Showing {len(evidence)} of {len(state.evidence)} total sources[/bold]\n")
        else:
            console.print(f"[bold]Total Evidence:[/bold] {len(evidence)} sources\n")

        # Summary statistics
        display_evidence_summary(state, console)

        # Evidence table
        if evidence:
            display_evidence_table(evidence, console)
        else:
            console.print("[dim]No evidence matches the specified filters.[/dim]")

        # Usage hints
        console.print("\n[dim]Tips:[/dim]")
        console.print(f"  [dim]• View details: --evidence-id E1[/dim]")
        console.print(f"  [dim]• Filter by type: --type forum[/dim]")
        console.print(f"  [dim]• Filter by credibility: --credibility high[/dim]")

        return 0

    except FileNotFoundError:
        log_error(f"Run not found: {run_id}")
        return 1
    except Exception as e:
        log_error(f"Error inspecting evidence: {e}")
        return 1


def display_evidence_summary(state: State, console: Console) -> None:
    """Display evidence summary statistics.

    Args:
        state: The state containing evidence
        console: Rich console instance
    """
    # Count by type
    by_type = {}
    for e in state.evidence:
        by_type[e.type] = by_type.get(e.type, 0) + 1

    # Count by credibility
    by_credibility = {}
    for e in state.evidence:
        by_credibility[e.credibility] = by_credibility.get(e.credibility, 0) + 1

    # Count by query category
    by_category = {}
    for e in state.evidence:
        for tag in e.tags:
            by_category[tag] = by_category.get(tag, 0) + 1

    # Summary table
    summary_table = Table(title="Evidence Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="white", width=20)
    summary_table.add_column("Breakdown", style="yellow")

    # Type breakdown
    type_str = ", ".join(f"{t}: {c}" for t, c in sorted(by_type.items(), key=lambda x: -x[1]))
    summary_table.add_row("By Type", type_str or "N/A")

    # Credibility breakdown with colors
    cred_parts = []
    for tier in ["high", "medium", "low"]:
        if tier in by_credibility:
            color = {"high": "green", "medium": "yellow", "low": "dim"}[tier]
            cred_parts.append(f"[{color}]{tier}: {by_credibility[tier]}[/{color}]")
    summary_table.add_row("By Credibility", ", ".join(cred_parts) or "N/A")

    # Category breakdown
    cat_str = ", ".join(f"{c}: {n}" for c, n in sorted(by_category.items(), key=lambda x: -x[1]))
    summary_table.add_row("By Category", cat_str or "N/A")

    # Query count
    queries_done = len([q for q in state.research_plan.queries if q.status == "done"])
    summary_table.add_row("Queries Executed", f"{queries_done} of {len(state.research_plan.queries)}")

    console.print(summary_table)
    console.print()


def display_evidence_table(evidence: list, console: Console) -> None:
    """Display evidence in a table format.

    Args:
        evidence: List of evidence items
        console: Rich console instance
    """
    table = Table(title="Evidence List", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=5, no_wrap=True)
    table.add_column("Type", style="white", width=9)
    table.add_column("Cred", width=8)
    table.add_column("Title", style="white", no_wrap=False)
    table.add_column("Tags", style="dim", width=12)

    for e in evidence:
        # Color code credibility
        cred_color = {"high": "green", "medium": "yellow", "low": "dim"}[e.credibility]
        cred_display = f"[{cred_color}]{e.credibility}[/{cred_color}]"

        # Truncate title
        title = e.title[:55] + "..." if len(e.title) > 58 else e.title

        # Tags
        tags = ", ".join(e.tags[:2]) if e.tags else ""

        table.add_row(
            e.id,
            e.type,
            cred_display,
            title,
            tags
        )

    console.print(table)


def display_evidence_detail(state: State, evidence_id: str, console: Console) -> int:
    """Display full details for a specific evidence item.

    Args:
        state: The state containing evidence
        evidence_id: The evidence ID to display
        console: Rich console instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Find the evidence
    evidence = None
    for e in state.evidence:
        if e.id == evidence_id or e.id.lower() == evidence_id.lower():
            evidence = e
            break

    if not evidence:
        log_error(f"Evidence not found: {evidence_id}")
        console.print(f"\n[dim]Available IDs: {', '.join(e.id for e in state.evidence[:20])}{'...' if len(state.evidence) > 20 else ''}[/dim]")
        return 1

    log_section(f"Evidence Detail: {evidence.id}", console)

    # Basic info table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Field", style="bold cyan", width=15)
    info_table.add_column("Value", style="white")

    info_table.add_row("ID", evidence.id)
    info_table.add_row("Title", evidence.title)
    info_table.add_row("URL", evidence.url)
    info_table.add_row("Type", evidence.type)

    cred_color = {"high": "green", "medium": "yellow", "low": "dim"}[evidence.credibility]
    info_table.add_row("Credibility", f"[{cred_color}]{evidence.credibility}[/{cred_color}]")

    info_table.add_row("Relevance", f"{evidence.relevance_score:.2f}")
    info_table.add_row("Tags", ", ".join(evidence.tags) if evidence.tags else "None")
    info_table.add_row("Query ID", evidence.query_id)

    console.print(info_table)

    # Find the query
    query_text = "Unknown"
    for q in state.research_plan.queries:
        if q.id == evidence.query_id:
            query_text = q.text
            break

    console.print(f"\n[bold]Source Query:[/bold] {query_text}")

    # Snippet
    console.print(f"\n[bold]Snippet:[/bold]")
    console.print(f"[dim]{evidence.snippet}[/dim]")

    # Full text preview
    if evidence.full_text:
        console.print(f"\n[bold]Content Preview:[/bold] ({len(evidence.full_text)} chars)")
        preview = evidence.full_text[:1000]
        if len(evidence.full_text) > 1000:
            preview += "\n\n[dim]... (truncated)[/dim]"
        console.print(f"{preview}")

    console.print()
    return 0


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

    # Handle --inspect
    if args.inspect:
        return inspect_evidence(
            args.inspect,
            console,
            type_filter=args.type,
            credibility_filter=args.credibility,
            evidence_id=args.evidence_id
        )

    # Handle --resume
    if args.resume:
        return resume_run(args.resume, console, verbose=args.verbose)

    # Handle new run
    if args.idea:
        return create_new_run(args.idea, console, verbose=args.verbose)

    # No valid arguments provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
