"""
Test full analysis pipeline: Research → Pain Points → Competitors
"""

from app.state import create_new_state, save_state
from app.config import get_config
from app.orchestrator import Orchestrator
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

console = Console()

# Test idea - use something with good forum/review content
idea = "Build a SaaS tool for freelance designers to manage client invoices, track payments, and handle tax reporting"

console.print(f"\n[bold blue]Full Analysis Pipeline Test[/bold blue]")
console.print(f"Idea: {idea}\n")

# Create state
state = create_new_state(idea)
config = get_config()
client = OpenAI(api_key=config.openai_api_key)

# Run orchestrator
console.print("[yellow]Running full pipeline (this may take 5-10 minutes)...[/yellow]\n")
orchestrator = Orchestrator(client)
state = orchestrator.run(state)

# Display results
console.print("\n[bold green]✓ Analysis Pipeline Complete![/bold green]\n")

# Task board
task_table = Table(title="Task Board")
task_table.add_column("Task", style="cyan", width=40)
task_table.add_column("Status", style="yellow", width=10)

for task in state.task_board:
    status_color = {"done": "green", "pending": "dim", "doing": "yellow", "blocked": "red"}[task.status]
    task_table.add_row(
        task.description,
        f"[{status_color}]{task.status}[/{status_color}]"
    )

console.print(task_table)
console.print()

# Summary stats
console.print(Panel.fit(
    f"[bold]Evidence:[/bold] {len(state.evidence)} sources\n"
    f"[bold]Pain Points:[/bold] {len(state.insights.pain_points)} clusters\n"
    f"[bold]Competitors:[/bold] {len(state.insights.competitors)} analyzed\n"
    f"[bold]Opportunity Gaps:[/bold] {len(state.insights.opportunity_gaps)} identified",
    title="Analysis Summary",
    border_style="green"
))
console.print()

# Pain Points Table
if state.insights.pain_points:
    pp_table = Table(title="Pain Point Clusters")
    pp_table.add_column("#", width=3)
    pp_table.add_column("Cluster", style="cyan", width=30)
    pp_table.add_column("Severity", style="yellow", width=8)
    pp_table.add_column("Who", style="white", width=40)

    for i, pp in enumerate(state.insights.pain_points[:10], 1):
        # Handle both Pydantic model and dict access
        if hasattr(pp, 'severity'):
            severity = pp.severity
            cluster_name = pp.cluster_name or "Unknown"
            who = pp.who or "Unknown"
        else:
            severity = pp.get("severity", "medium")
            cluster_name = pp.get("cluster_name", "Unknown")
            who = pp.get("who", "Unknown")

        sev_color = {"critical": "red", "major": "yellow", "minor": "dim", "high": "red", "medium": "yellow", "low": "dim"}.get(severity, "dim")
        pp_table.add_row(
            str(i),
            cluster_name[:27] + "..." if len(cluster_name) > 30 else cluster_name,
            f"[{sev_color}]{severity}[/{sev_color}]",
            who[:37] + "..." if len(who) > 40 else who
        )

    console.print(pp_table)
    console.print()

# Competitors Table
if state.insights.competitors:
    comp_table = Table(title="Competitive Landscape")
    comp_table.add_column("#", width=3)
    comp_table.add_column("Competitor", style="cyan", width=20)
    comp_table.add_column("Positioning", style="white", width=50)
    comp_table.add_column("Features", style="dim", width=8)

    for i, comp in enumerate(state.insights.competitors[:10], 1):
        # Handle both Pydantic model and dict access
        if hasattr(comp, 'name'):
            name = comp.name
            positioning = comp.positioning or comp.description or "N/A"
            features_count = len(comp.key_features) if comp.key_features else 0
        else:
            name = comp.get("name", "Unknown")
            positioning = comp.get("positioning", comp.get("description", "N/A"))
            features_count = len(comp.get("key_features", []))

        comp_table.add_row(
            str(i),
            name[:17] + "..." if len(name) > 20 else name,
            positioning[:47] + "..." if len(positioning) > 50 else positioning,
            str(features_count)
        )

    console.print(comp_table)
    console.print()

# Opportunity Gaps
if state.insights.opportunity_gaps:
    console.print("[bold cyan]Opportunity Gaps:[/bold cyan]")
    for i, gap in enumerate(state.insights.opportunity_gaps, 1):
        console.print(f"  {i}. {gap}")
    console.print()

# Market Insights
if state.insights.market_insights:
    console.print(Panel.fit(
        state.insights.market_insights,
        title="Market Insights",
        border_style="blue"
    ))
    console.print()

# Save state
save_state(state, state.run_id)
console.print(f"[dim]State saved: data/runs/{state.run_id}.json[/dim]")

# Save detailed analysis report
def to_dict(obj):
    """Convert Pydantic model or dict to dict."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return dict(obj.__dict__)
    return obj

analysis_report = {
    "run_id": state.run_id,
    "idea": idea,
    "metadata": {
        "domain": state.metadata.domain,
        "target_user": state.metadata.target_user,
        "geography": state.metadata.geography,
    },
    "evidence_summary": {
        "total": len(state.evidence),
        "by_type": {},
        "by_credibility": {},
    },
    "pain_points": [to_dict(pp) for pp in state.insights.pain_points],
    "competitors": [to_dict(c) for c in state.insights.competitors],
    "opportunity_gaps": state.insights.opportunity_gaps,
    "market_insights": state.insights.market_insights,
}

# Calculate evidence breakdown
for e in state.evidence:
    e_dict = to_dict(e)
    e_type = e_dict.get("type", "unknown")
    e_cred = e_dict.get("credibility", "unknown")
    analysis_report["evidence_summary"]["by_type"][e_type] = \
        analysis_report["evidence_summary"]["by_type"].get(e_type, 0) + 1
    analysis_report["evidence_summary"]["by_credibility"][e_cred] = \
        analysis_report["evidence_summary"]["by_credibility"].get(e_cred, 0) + 1

with open(f"data/analysis_report_{state.run_id}.json", "w") as f:
    json.dump(analysis_report, f, indent=2)

console.print(f"[dim]Analysis report: data/analysis_report_{state.run_id}.json[/dim]\n")

# Show sample pain point detail
if state.insights.pain_points:
    console.print("[bold cyan]Sample Pain Point Detail:[/bold cyan]\n")
    pp = state.insights.pain_points[0]

    # Handle both Pydantic model and dict access
    if hasattr(pp, 'cluster_name'):
        console.print(f"[bold]Cluster:[/bold] {pp.cluster_name}")
        console.print(f"[bold]Who:[/bold] {pp.who}")
        console.print(f"[bold]What:[/bold] {pp.what}")
        console.print(f"[bold]Why:[/bold] {pp.why}")
        console.print(f"[bold]Severity:[/bold] {pp.severity}")
        console.print(f"[bold]Evidence:[/bold] {', '.join(pp.evidence_ids)}")
        console.print(f"[bold]Example Quotes:[/bold]")
        for quote in (pp.example_quotes or [])[:2]:
            console.print(f"  • \"{quote}\"")
    else:
        console.print(f"[bold]Cluster:[/bold] {pp.get('cluster_name', 'N/A')}")
        console.print(f"[bold]Who:[/bold] {pp.get('who', 'N/A')}")
        console.print(f"[bold]What:[/bold] {pp.get('what', 'N/A')}")
        console.print(f"[bold]Why:[/bold] {pp.get('why', 'N/A')}")
        console.print(f"[bold]Severity:[/bold] {pp.get('severity', 'N/A')}")
        console.print(f"[bold]Evidence:[/bold] {', '.join(pp.get('evidence_ids', []))}")
        console.print(f"[bold]Example Quotes:[/bold]")
        for quote in pp.get('example_quotes', [])[:2]:
            console.print(f"  • \"{quote}\"")
    console.print()

console.print("[green]✓ Full analysis pipeline test complete![/green]\n")
