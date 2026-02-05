"""Main CLI interface for the multi-agent PRD generator.

This module provides the command-line interface for creating new PRD generation
runs, resuming existing runs, and listing all runs.
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
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

  # Inspect analysis results
  python -m app.main --inspect <run_id> --painpoints
  python -m app.main --inspect <run_id> --competitors
  python -m app.main --inspect <run_id> --gaps

  # Export PRD
  python -m app.main --export <run_id>
  python -m app.main --export <run_id> --format json
  python -m app.main --export <run_id> --format markdown
  python -m app.main --export <run_id> --output-dir ./my-exports/
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

    # Analysis inspection options
    parser.add_argument(
        "--painpoints",
        action="store_true",
        help="Show pain points analysis (use with --inspect)"
    )

    parser.add_argument(
        "--painpoint-id",
        type=str,
        metavar="ID",
        help="Show full details for specific pain point (use with --inspect --painpoints)"
    )

    parser.add_argument(
        "--competitors",
        action="store_true",
        help="Show competitive analysis (use with --inspect)"
    )

    parser.add_argument(
        "--competitor-id",
        type=str,
        metavar="ID",
        help="Show full details for specific competitor (use with --inspect --competitors)"
    )

    parser.add_argument(
        "--gaps",
        action="store_true",
        help="Show opportunity gaps and market insights (use with --inspect)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed agent trace and execution logs"
    )

    # Export options
    parser.add_argument(
        "--export",
        type=str,
        metavar="RUN_ID",
        help="Export PRD from a completed run"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "both"],
        default="both",
        help="Export format: markdown, json, or both (default: both)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        help="Output directory for exports (default: data/outputs/)"
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


def inspect_painpoints(
    run_id: str,
    console: Console,
    painpoint_id: str = None
) -> int:
    """Inspect pain points analysis from a run.

    Args:
        run_id: The run ID to inspect
        console: Rich console instance for output
        painpoint_id: Optional specific pain point ID to show details

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        state = load_state(run_id)

        if not state.insights.pain_points:
            log_info(f"No pain points found for run {run_id[:8]}...")
            log_info("Pain points are extracted after the research phase completes.")
            return 0

        # If specific pain point requested
        if painpoint_id:
            return display_painpoint_detail(state, painpoint_id, console)

        # Display summary
        log_section(f"Pain Points Analysis: {run_id[:8]}...", console)

        console.print(f"[bold]Product Idea:[/bold] {state.metadata.raw_idea[:80]}...")
        console.print(f"[bold]Target User:[/bold] {state.metadata.target_user}")
        console.print(f"[bold]Domain:[/bold] {state.metadata.domain}")
        console.print()

        # Summary panel
        severity_counts = {"critical": 0, "major": 0, "minor": 0}
        for pp in state.insights.pain_points:
            sev = pp.severity if hasattr(pp, 'severity') else pp.get('severity', 'major')
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        console.print(Panel.fit(
            f"[bold]Total Pain Points:[/bold] {len(state.insights.pain_points)}\n"
            f"[red]Critical:[/red] {severity_counts.get('critical', 0)}  "
            f"[yellow]Major:[/yellow] {severity_counts.get('major', 0)}  "
            f"[dim]Minor:[/dim] {severity_counts.get('minor', 0)}",
            title="Pain Points Summary",
            border_style="cyan"
        ))
        console.print()

        # Pain points table
        table = Table(title="Pain Point Clusters", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=5)
        table.add_column("Cluster", style="white", width=28)
        table.add_column("Sev", width=8)
        table.add_column("Who", style="dim", width=35)
        table.add_column("Evidence", style="dim", width=10)

        for pp in state.insights.pain_points:
            # Handle both Pydantic model and dict
            if hasattr(pp, 'id'):
                pp_id = pp.id or "N/A"
                cluster = pp.cluster_name or pp.description[:25]
                severity = pp.severity
                who = pp.who or "N/A"
                evidence_ids = pp.evidence_ids or []
            else:
                pp_id = pp.get('id', 'N/A')
                cluster = pp.get('cluster_name', pp.get('description', '')[:25])
                severity = pp.get('severity', 'major')
                who = pp.get('who', 'N/A')
                evidence_ids = pp.get('evidence_ids', [])

            sev_color = {"critical": "red", "major": "yellow", "minor": "dim"}.get(severity, "white")
            cluster_display = cluster[:25] + "..." if len(cluster) > 28 else cluster
            who_display = who[:32] + "..." if len(who) > 35 else who

            table.add_row(
                pp_id,
                cluster_display,
                f"[{sev_color}]{severity}[/{sev_color}]",
                who_display,
                f"{len(evidence_ids)} sources"
            )

        console.print(table)

        # Usage hints
        console.print("\n[dim]Tips:[/dim]")
        console.print(f"  [dim]• View details: --painpoint-id PP1[/dim]")
        console.print(f"  [dim]• View competitors: --competitors[/dim]")
        console.print(f"  [dim]• View gaps: --gaps[/dim]")

        return 0

    except FileNotFoundError:
        log_error(f"Run not found: {run_id}")
        return 1
    except Exception as e:
        log_error(f"Error inspecting pain points: {e}")
        return 1


def display_painpoint_detail(state: State, painpoint_id: str, console: Console) -> int:
    """Display full details for a specific pain point.

    Args:
        state: The state containing pain points
        painpoint_id: The pain point ID to display
        console: Rich console instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Find the pain point
    painpoint = None
    for pp in state.insights.pain_points:
        pp_id = pp.id if hasattr(pp, 'id') else pp.get('id', '')
        if pp_id == painpoint_id or pp_id.lower() == painpoint_id.lower():
            painpoint = pp
            break

    if not painpoint:
        log_error(f"Pain point not found: {painpoint_id}")
        available = [pp.id if hasattr(pp, 'id') else pp.get('id', '') for pp in state.insights.pain_points]
        console.print(f"\n[dim]Available IDs: {', '.join(available)}[/dim]")
        return 1

    # Extract fields (handle both Pydantic and dict)
    if hasattr(painpoint, 'cluster_name'):
        cluster_name = painpoint.cluster_name
        who = painpoint.who
        what = painpoint.what
        why = painpoint.why
        severity = painpoint.severity
        frequency = painpoint.frequency
        quotes = painpoint.example_quotes or []
        evidence_ids = painpoint.evidence_ids or []
    else:
        cluster_name = painpoint.get('cluster_name', 'N/A')
        who = painpoint.get('who', 'N/A')
        what = painpoint.get('what', 'N/A')
        why = painpoint.get('why', 'N/A')
        severity = painpoint.get('severity', 'N/A')
        frequency = painpoint.get('frequency', 'N/A')
        quotes = painpoint.get('example_quotes', [])
        evidence_ids = painpoint.get('evidence_ids', [])

    log_section(f"Pain Point Detail: {painpoint_id}", console)

    # Info panel
    sev_color = {"critical": "red", "major": "yellow", "minor": "dim"}.get(severity, "white")

    console.print(Panel.fit(
        f"[bold cyan]Cluster:[/bold cyan] {cluster_name}\n"
        f"[bold]Severity:[/bold] [{sev_color}]{severity}[/{sev_color}]  "
        f"[bold]Frequency:[/bold] {frequency}",
        border_style="cyan"
    ))
    console.print()

    # Detail table
    detail_table = Table(show_header=False, box=None, padding=(0, 2))
    detail_table.add_column("Field", style="bold cyan", width=12)
    detail_table.add_column("Value", style="white")

    detail_table.add_row("Who", who or "N/A")
    detail_table.add_row("What", what or "N/A")
    detail_table.add_row("Why", why or "N/A")
    detail_table.add_row("Evidence", ", ".join(evidence_ids) if evidence_ids else "None")

    console.print(detail_table)

    # Example quotes
    if quotes:
        console.print("\n[bold cyan]Example Quotes:[/bold cyan]")
        for i, quote in enumerate(quotes, 1):
            console.print(f"  {i}. [italic]\"{quote}\"[/italic]")

    # Show linked evidence
    if evidence_ids:
        console.print("\n[bold cyan]Linked Evidence:[/bold cyan]")
        for eid in evidence_ids:
            for e in state.evidence:
                e_id = e.id if hasattr(e, 'id') else e.get('id', '')
                if e_id == eid:
                    title = e.title if hasattr(e, 'title') else e.get('title', '')
                    e_type = e.type if hasattr(e, 'type') else e.get('type', '')
                    console.print(f"  [{eid}] {title[:60]}{'...' if len(title) > 60 else ''} ({e_type})")
                    break

    console.print()
    return 0


def inspect_competitors(
    run_id: str,
    console: Console,
    competitor_id: str = None
) -> int:
    """Inspect competitive analysis from a run.

    Args:
        run_id: The run ID to inspect
        console: Rich console instance for output
        competitor_id: Optional specific competitor ID to show details

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        state = load_state(run_id)

        if not state.insights.competitors:
            log_info(f"No competitors found for run {run_id[:8]}...")
            log_info("Competitor analysis runs after the research phase completes.")
            return 0

        # If specific competitor requested
        if competitor_id:
            return display_competitor_detail(state, competitor_id, console)

        # Display summary
        log_section(f"Competitive Analysis: {run_id[:8]}...", console)

        console.print(f"[bold]Product Idea:[/bold] {state.metadata.raw_idea[:80]}...")
        console.print(f"[bold]Domain:[/bold] {state.metadata.domain}")
        console.print()

        # Summary panel
        console.print(Panel.fit(
            f"[bold]Competitors Analyzed:[/bold] {len(state.insights.competitors)}\n"
            f"[bold]Opportunity Gaps:[/bold] {len(state.insights.opportunity_gaps)}",
            title="Competitive Landscape Summary",
            border_style="cyan"
        ))
        console.print()

        # Competitors table
        table = Table(title="Competitors", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Name", style="white", width=18)
        table.add_column("Positioning", style="dim", width=40)
        table.add_column("Pricing", style="yellow", width=15)
        table.add_column("Features", style="dim", width=8)

        for comp in state.insights.competitors:
            # Handle both Pydantic model and dict
            if hasattr(comp, 'id'):
                comp_id = comp.id or "N/A"
                name = comp.name
                positioning = comp.positioning or comp.description or "N/A"
                pricing = comp.pricing_model or comp.pricing or "N/A"
                features = comp.key_features or []
            else:
                comp_id = comp.get('id', 'N/A')
                name = comp.get('name', 'Unknown')
                positioning = comp.get('positioning', comp.get('description', 'N/A'))
                pricing = comp.get('pricing_model', comp.get('pricing', 'N/A'))
                features = comp.get('key_features', [])

            pos_display = positioning[:37] + "..." if len(positioning) > 40 else positioning
            pricing_display = pricing[:12] + "..." if len(str(pricing)) > 15 else str(pricing)

            table.add_row(
                comp_id,
                name[:15] + "..." if len(name) > 18 else name,
                pos_display,
                pricing_display,
                str(len(features))
            )

        console.print(table)

        # Quick strengths/weaknesses summary
        console.print("\n[bold cyan]Quick Comparison:[/bold cyan]")
        for comp in state.insights.competitors[:5]:
            if hasattr(comp, 'name'):
                name = comp.name
                strengths = comp.strengths or []
                weaknesses = comp.weaknesses or []
            else:
                name = comp.get('name', 'Unknown')
                strengths = comp.get('strengths', [])
                weaknesses = comp.get('weaknesses', [])

            console.print(f"\n  [bold]{name}[/bold]")
            if strengths:
                console.print(f"    [green]+[/green] {strengths[0][:60]}")
            if weaknesses:
                console.print(f"    [red]-[/red] {weaknesses[0][:60]}")

        # Usage hints
        console.print("\n[dim]Tips:[/dim]")
        console.print(f"  [dim]• View details: --competitor-id C1[/dim]")
        console.print(f"  [dim]• View gaps: --gaps[/dim]")
        console.print(f"  [dim]• View pain points: --painpoints[/dim]")

        return 0

    except FileNotFoundError:
        log_error(f"Run not found: {run_id}")
        return 1
    except Exception as e:
        log_error(f"Error inspecting competitors: {e}")
        return 1


def display_competitor_detail(state: State, competitor_id: str, console: Console) -> int:
    """Display full details for a specific competitor.

    Args:
        state: The state containing competitors
        competitor_id: The competitor ID to display
        console: Rich console instance

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Find the competitor
    competitor = None
    for comp in state.insights.competitors:
        comp_id = comp.id if hasattr(comp, 'id') else comp.get('id', '')
        if comp_id == competitor_id or comp_id.lower() == competitor_id.lower():
            competitor = comp
            break

    if not competitor:
        log_error(f"Competitor not found: {competitor_id}")
        available = [c.id if hasattr(c, 'id') else c.get('id', '') for c in state.insights.competitors]
        console.print(f"\n[dim]Available IDs: {', '.join(available)}[/dim]")
        return 1

    # Extract fields (handle both Pydantic and dict)
    if hasattr(competitor, 'name'):
        name = competitor.name
        url = competitor.url
        positioning = competitor.positioning or competitor.description
        icp = competitor.icp
        pricing_model = competitor.pricing_model
        pricing_details = competitor.pricing_details or competitor.pricing
        features = competitor.key_features or []
        strengths = competitor.strengths or []
        weaknesses = competitor.weaknesses or []
        evidence_ids = competitor.evidence_ids or []
    else:
        name = competitor.get('name', 'Unknown')
        url = competitor.get('url')
        positioning = competitor.get('positioning', competitor.get('description', 'N/A'))
        icp = competitor.get('icp', 'N/A')
        pricing_model = competitor.get('pricing_model', 'N/A')
        pricing_details = competitor.get('pricing_details', competitor.get('pricing', 'N/A'))
        features = competitor.get('key_features', [])
        strengths = competitor.get('strengths', [])
        weaknesses = competitor.get('weaknesses', [])
        evidence_ids = competitor.get('evidence_ids', [])

    log_section(f"Competitor Detail: {name}", console)

    # Header panel
    console.print(Panel.fit(
        f"[bold cyan]{name}[/bold cyan]\n"
        f"[dim]{url or 'No URL available'}[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Basic info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Field", style="bold cyan", width=15)
    info_table.add_column("Value", style="white")

    info_table.add_row("Positioning", positioning or "N/A")
    info_table.add_row("Target Customer", icp or "N/A")
    info_table.add_row("Pricing Model", pricing_model or "N/A")
    info_table.add_row("Pricing Details", pricing_details or "N/A")
    info_table.add_row("Evidence", ", ".join(evidence_ids) if evidence_ids else "None")

    console.print(info_table)

    # Key Features
    if features:
        console.print("\n[bold cyan]Key Features:[/bold cyan]")
        for i, feature in enumerate(features, 1):
            console.print(f"  {i}. {feature}")

    # Strengths and Weaknesses side by side
    console.print()
    sw_table = Table(show_header=True, header_style="bold", box=None)
    sw_table.add_column("[green]Strengths[/green]", style="green", width=40)
    sw_table.add_column("[red]Weaknesses[/red]", style="red", width=40)

    max_items = max(len(strengths), len(weaknesses))
    for i in range(max_items):
        s = strengths[i] if i < len(strengths) else ""
        w = weaknesses[i] if i < len(weaknesses) else ""
        sw_table.add_row(f"+ {s}" if s else "", f"- {w}" if w else "")

    console.print(sw_table)

    # Show linked evidence
    if evidence_ids:
        console.print("\n[bold cyan]Linked Evidence:[/bold cyan]")
        for eid in evidence_ids[:5]:
            for e in state.evidence:
                e_id = e.id if hasattr(e, 'id') else e.get('id', '')
                if e_id == eid:
                    title = e.title if hasattr(e, 'title') else e.get('title', '')
                    e_type = e.type if hasattr(e, 'type') else e.get('type', '')
                    console.print(f"  [{eid}] {title[:55]}{'...' if len(title) > 55 else ''} ({e_type})")
                    break
        if len(evidence_ids) > 5:
            console.print(f"  [dim]... and {len(evidence_ids) - 5} more[/dim]")

    console.print()
    return 0


def inspect_gaps(run_id: str, console: Console) -> int:
    """Inspect opportunity gaps and market insights from a run.

    Args:
        run_id: The run ID to inspect
        console: Rich console instance for output

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        state = load_state(run_id)

        log_section(f"Opportunity Analysis: {run_id[:8]}...", console)

        console.print(f"[bold]Product Idea:[/bold] {state.metadata.raw_idea[:80]}...")
        console.print(f"[bold]Domain:[/bold] {state.metadata.domain}")
        console.print()

        # Check if we have gaps
        if not state.insights.opportunity_gaps and not state.insights.market_insights:
            log_info("No opportunity gaps or market insights found.")
            log_info("This analysis runs after the competitor analysis completes.")
            return 0

        # Summary panel
        console.print(Panel.fit(
            f"[bold]Opportunity Gaps:[/bold] {len(state.insights.opportunity_gaps)}\n"
            f"[bold]Competitors Analyzed:[/bold] {len(state.insights.competitors)}\n"
            f"[bold]Pain Points Identified:[/bold] {len(state.insights.pain_points)}",
            title="Analysis Summary",
            border_style="green"
        ))
        console.print()

        # Opportunity Gaps
        if state.insights.opportunity_gaps:
            console.print("[bold cyan]Opportunity Gaps:[/bold cyan]")
            console.print("[dim]What competitors collectively miss or do poorly:[/dim]\n")

            for i, gap in enumerate(state.insights.opportunity_gaps, 1):
                console.print(f"  [bold green]{i}.[/bold green] {gap}")

            console.print()

        # Market Insights
        if state.insights.market_insights:
            console.print(Panel(
                state.insights.market_insights,
                title="[bold]Market Insights[/bold]",
                border_style="blue",
                padding=(1, 2)
            ))
            console.print()

        # Cross-reference with pain points
        if state.insights.pain_points:
            console.print("[bold cyan]Related Pain Points:[/bold cyan]")
            console.print("[dim]User pain points that align with opportunity gaps:[/dim]\n")

            # Show top 5 high-severity pain points
            high_severity = []
            for pp in state.insights.pain_points:
                sev = pp.severity if hasattr(pp, 'severity') else pp.get('severity', '')
                if sev in ['critical', 'high']:
                    high_severity.append(pp)

            for pp in high_severity[:5]:
                if hasattr(pp, 'cluster_name'):
                    cluster = pp.cluster_name
                    what = pp.what
                else:
                    cluster = pp.get('cluster_name', pp.get('description', ''))
                    what = pp.get('what', '')

                console.print(f"  [red]•[/red] [bold]{cluster}[/bold]")
                if what:
                    console.print(f"    [dim]{what[:80]}{'...' if len(what) > 80 else ''}[/dim]")

        # Cross-reference with competitor weaknesses
        if state.insights.competitors:
            console.print("\n[bold cyan]Common Competitor Weaknesses:[/bold cyan]")
            console.print("[dim]Weaknesses mentioned across multiple competitors:[/dim]\n")

            # Collect all weaknesses
            all_weaknesses = []
            for comp in state.insights.competitors:
                weaknesses = comp.weaknesses if hasattr(comp, 'weaknesses') else comp.get('weaknesses', [])
                name = comp.name if hasattr(comp, 'name') else comp.get('name', '')
                for w in weaknesses:
                    all_weaknesses.append((name, w))

            # Show unique weaknesses
            shown = set()
            for name, weakness in all_weaknesses[:8]:
                w_lower = weakness.lower()[:30]
                if w_lower not in shown:
                    shown.add(w_lower)
                    console.print(f"  [yellow]•[/yellow] {weakness[:70]}{'...' if len(weakness) > 70 else ''}")
                    console.print(f"    [dim]({name})[/dim]")

        # Usage hints
        console.print("\n[dim]Tips:[/dim]")
        console.print(f"  [dim]• View pain points: --painpoints[/dim]")
        console.print(f"  [dim]• View competitors: --competitors[/dim]")
        console.print(f"  [dim]• View evidence: (no flag, just --inspect)[/dim]")

        return 0

    except FileNotFoundError:
        log_error(f"Run not found: {run_id}")
        return 1
    except Exception as e:
        log_error(f"Error inspecting gaps: {e}")
        return 1


def export_prd(
    run_id: str,
    console: Console,
    format: str = "both",
    output_dir: str = None
) -> int:
    """Export PRD from a completed run.

    Args:
        run_id: The run ID to export
        console: Rich console instance for output
        format: Export format (markdown, json, or both)
        output_dir: Output directory (default: data/outputs/)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    import json
    import os

    try:
        # Load state
        state = load_state(run_id)

        log_section(f"Exporting PRD: {run_id[:8]}...", console)

        # Check if PRD exists
        if not state.prd.notion_markdown and not state.prd.sections:
            log_error("No PRD found in this run. Make sure the PRD generation completed.")
            return 1

        # Set output directory
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = Path("data/outputs")

        # Create output directory if needed
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate filenames
        product_name = state.prd.sections.get("product_name", "PRD")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in product_name)
        safe_name = safe_name.replace(" ", "_")[:30]

        md_filename = f"{run_id[:8]}_{safe_name}_PRD.md"
        json_filename = f"{run_id[:8]}_{safe_name}_PRD.json"

        exported_files = []
        total_size = 0

        # Export Markdown
        if format in ["markdown", "both"]:
            md_path = out_dir / md_filename

            if state.prd.notion_markdown:
                md_content = state.prd.notion_markdown
            else:
                # Generate basic markdown from sections if notion_markdown not available
                md_content = _generate_basic_markdown(state)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            md_size = os.path.getsize(md_path)
            total_size += md_size
            exported_files.append({
                "type": "Markdown",
                "path": str(md_path),
                "size": md_size
            })

        # Export JSON
        if format in ["json", "both"]:
            json_path = out_dir / json_filename

            # Build comprehensive JSON export
            json_export = {
                "run_id": state.run_id,
                "created_at": state.created_at,
                "status": state.status,
                "metadata": {
                    "raw_idea": state.metadata.raw_idea,
                    "domain": state.metadata.domain,
                    "target_user": state.metadata.target_user,
                    "geography": state.metadata.geography,
                    "industry_tags": state.metadata.industry_tags,
                    "compliance_contexts": state.metadata.compliance_contexts,
                },
                "prd": {
                    "sections": state.prd.sections,
                    "citation_map": state.prd.citation_map,
                },
                "insights": {
                    "pain_points": [
                        pp.model_dump() if hasattr(pp, 'model_dump') else pp
                        for pp in state.insights.pain_points
                    ],
                    "competitors": [
                        comp.model_dump() if hasattr(comp, 'model_dump') else comp
                        for comp in state.insights.competitors
                    ],
                    "opportunity_gaps": state.insights.opportunity_gaps,
                    "market_insights": state.insights.market_insights,
                },
                "quality_report": state.quality_report,
                "evidence_summary": {
                    "total_sources": len(state.evidence),
                    "by_type": _count_by_field(state.evidence, "type"),
                    "by_credibility": _count_by_field(state.evidence, "credibility"),
                },
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_export, f, indent=2, ensure_ascii=False)

            json_size = os.path.getsize(json_path)
            total_size += json_size
            exported_files.append({
                "type": "JSON",
                "path": str(json_path),
                "size": json_size
            })

        # Get quality metrics
        quality_score = state.quality_report.get("quality_score", "N/A") if state.quality_report else "N/A"
        citation_coverage = state.quality_report.get("citation_coverage_pct", "N/A") if state.quality_report else "N/A"
        passed = state.quality_report.get("passed", False) if state.quality_report else False

        # Count sections
        section_count = len([k for k, v in state.prd.sections.items() if v]) if state.prd.sections else 0

        # Display results
        _display_export_results(
            console=console,
            exported_files=exported_files,
            total_size=total_size,
            product_name=product_name,
            quality_score=quality_score,
            citation_coverage=citation_coverage,
            passed=passed,
            section_count=section_count,
            evidence_count=len(state.evidence),
            pain_points_count=len(state.insights.pain_points),
            competitors_count=len(state.insights.competitors),
        )

        return 0

    except FileNotFoundError:
        log_error(f"Run not found: {run_id}")
        return 1
    except Exception as e:
        log_error(f"Error exporting PRD: {e}")
        return 1


def _count_by_field(items: list, field: str) -> dict:
    """Count items by a specific field.

    Args:
        items: List of items
        field: Field to count by

    Returns:
        Dictionary of counts
    """
    counts = {}
    for item in items:
        if hasattr(item, field):
            value = getattr(item, field)
        elif isinstance(item, dict):
            value = item.get(field, "unknown")
        else:
            value = "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _generate_basic_markdown(state: State) -> str:
    """Generate basic markdown from PRD sections.

    Args:
        state: The state containing PRD sections

    Returns:
        Markdown string
    """
    sections = state.prd.sections
    lines = []

    lines.append(f"# {sections.get('product_name', 'Product Requirements Document')}")
    lines.append("")

    if sections.get("problem_statement"):
        lines.append("## Problem Statement")
        lines.append(sections["problem_statement"])
        lines.append("")

    if sections.get("target_users"):
        lines.append("## Target Users")
        lines.append(sections["target_users"])
        lines.append("")

    if sections.get("solution_overview"):
        lines.append("## Solution Overview")
        lines.append(sections["solution_overview"])
        lines.append("")

    if sections.get("value_proposition"):
        lines.append("## Value Proposition")
        lines.append(sections["value_proposition"])
        lines.append("")

    if sections.get("mvp_features"):
        lines.append("## MVP Features")
        for feature in sections["mvp_features"]:
            lines.append(f"- {feature}")
        lines.append("")

    if sections.get("success_metrics"):
        lines.append("## Success Metrics")
        for metric in sections["success_metrics"]:
            lines.append(f"- {metric}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Run ID: {state.run_id}*")

    return "\n".join(lines)


def _display_export_results(
    console: Console,
    exported_files: list,
    total_size: int,
    product_name: str,
    quality_score,
    citation_coverage,
    passed: bool,
    section_count: int,
    evidence_count: int,
    pain_points_count: int,
    competitors_count: int,
) -> None:
    """Display export results with rich formatting.

    Args:
        console: Rich console instance
        exported_files: List of exported file info
        total_size: Total size in bytes
        product_name: Product name
        quality_score: Quality score
        citation_coverage: Citation coverage percentage
        passed: Whether validation passed
        section_count: Number of PRD sections
        evidence_count: Number of evidence items
        pain_points_count: Number of pain points
        competitors_count: Number of competitors
    """
    # Format file size
    def format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    # Success panel
    status_color = "green" if passed else "yellow"
    status_text = "PASSED" if passed else "NEEDS REVIEW"

    quality_display = f"{quality_score:.0f}/100" if isinstance(quality_score, (int, float)) else str(quality_score)
    citation_display = f"{citation_coverage:.0f}%" if isinstance(citation_coverage, (int, float)) else str(citation_coverage)

    console.print(Panel.fit(
        f"[bold green]Export Successful![/bold green]\n\n"
        f"[bold]Product:[/bold] {product_name}\n"
        f"[bold]Validation:[/bold] [{status_color}]{status_text}[/{status_color}]",
        title="PRD Export",
        border_style="green"
    ))
    console.print()

    # Files table
    files_table = Table(title="Exported Files", show_header=True, header_style="bold cyan")
    files_table.add_column("Format", style="yellow", width=12)
    files_table.add_column("Path", style="white")
    files_table.add_column("Size", style="green", justify="right", width=12)

    for f in exported_files:
        files_table.add_row(
            f["type"],
            f["path"],
            format_size(f["size"])
        )

    console.print(files_table)
    console.print(f"\n[bold]Total Size:[/bold] {format_size(total_size)}")
    console.print()

    # Quality metrics table
    metrics_table = Table(title="PRD Metrics", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan", width=25)
    metrics_table.add_column("Value", style="white", width=20)

    metrics_table.add_row("Quality Score", quality_display)
    metrics_table.add_row("Citation Coverage", citation_display)
    metrics_table.add_row("PRD Sections", str(section_count))
    metrics_table.add_row("Evidence Sources", str(evidence_count))
    metrics_table.add_row("Pain Points", str(pain_points_count))
    metrics_table.add_row("Competitors Analyzed", str(competitors_count))

    console.print(metrics_table)
    console.print()

    # Usage hints
    log_success("PRD exported successfully!")
    console.print("\n[dim]You can now:[/dim]")
    console.print(f"  [dim]• Open the Markdown file in any editor[/dim]")
    console.print(f"  [dim]• Import to Notion using the markdown[/dim]")
    console.print(f"  [dim]• Use the JSON for programmatic access[/dim]")


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

    # Handle --export
    if args.export:
        return export_prd(
            args.export,
            console,
            format=args.format,
            output_dir=args.output_dir
        )

    # Handle --inspect with various sub-options
    if args.inspect:
        # Pain points inspection
        if args.painpoints:
            return inspect_painpoints(
                args.inspect,
                console,
                painpoint_id=args.painpoint_id
            )

        # Competitors inspection
        if args.competitors:
            return inspect_competitors(
                args.inspect,
                console,
                competitor_id=args.competitor_id
            )

        # Opportunity gaps inspection
        if args.gaps:
            return inspect_gaps(args.inspect, console)

        # Default: evidence inspection
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
