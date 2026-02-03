"""Test script for PlannerAgent standalone execution."""

from openai import OpenAI

from app.state import create_new_state, save_state
from app.config import get_config
from agents.clarification import ClarificationAgent
from agents.planner import PlannerAgent
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()

# Test idea
idea = "Build a SaaS tool for customer support teams to manage tickets and track response times"

# Create state and run clarification
rprint(f"[bold blue]Testing PlannerAgent[/bold blue]")
rprint(f"Idea: {idea}\n")

state = create_new_state(idea)
config = get_config()
client = OpenAI(api_key=config.openai_api_key)

# Step 1: Clarification
rprint("[yellow]Step 1: Running ClarificationAgent...[/yellow]")
clarification_agent = ClarificationAgent("clarification", client)
state = clarification_agent.run(state)
state.metadata.clarification_status = "confirmed"

rprint(f"Domain: {state.metadata.domain}")
rprint(f"Target User: {state.metadata.target_user}\n")

# Step 2: Planning
rprint("[yellow]Step 2: Running PlannerAgent...[/yellow]")
planner_agent = PlannerAgent("planning", client)
state = planner_agent.run(state)

# Display results
rprint("\n[bold green]✓ Query Generation Complete[/bold green]\n")

# Summary stats
queries = state.research_plan.queries
total = len(queries)
by_category = {}
by_priority = {}

for q in queries:
    by_category[q.category] = by_category.get(q.category, 0) + 1
    by_priority[q.priority] = by_priority.get(q.priority, 0) + 1

rprint(f"[bold]Total Queries:[/bold] {total}")
rprint(f"[bold]By Category:[/bold] {by_category}")
rprint(f"[bold]By Priority:[/bold] {by_priority}\n")

# Show queries in table
table = Table(title="Generated Research Queries")
table.add_column("#", style="dim", width=3)
table.add_column("Category", style="cyan", width=12)
table.add_column("Priority", style="yellow", width=8)
table.add_column("Query", style="white", width=60)
table.add_column("Sources", style="dim", width=30)

for i, q in enumerate(queries, 1):
    priority_color = {"high": "red", "medium": "yellow", "low": "dim"}[q.priority]
    table.add_row(
        str(i),
        q.category,
        f"[{priority_color}]{q.priority}[/{priority_color}]",
        q.text[:57] + "..." if len(q.text) > 60 else q.text,
        ", ".join(q.expected_sources[:3])
    )

console.print(table)

# Show rationale from agent trace
rprint("\n[bold]Agent Trace:[/bold]")
for entry in state.agent_trace:
    if entry.agent == "planning":
        rprint(f"  [dim]{entry.turn}.[/dim] {entry.action}")

# Save state
save_state(state, state.run_id)
rprint(f"\n[dim]Saved to: data/runs/{state.run_id}.json[/dim]")
