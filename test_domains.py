"""
Test PlannerAgent across multiple domains
"""
from openai import OpenAI

from app.state import create_new_state, save_state
from app.config import get_config
from app.orchestrator import Orchestrator
from rich.console import Console
from rich.table import Table
import json

console = Console()
config = get_config()
client = OpenAI(api_key=config.openai_api_key)
orchestrator = Orchestrator(client, console=console)

TEST_IDEAS = [
    "Build a tool for freelance developers to track time and invoice clients",
    "HIPAA-compliant telemedicine platform for rural healthcare providers",
    "DevSecOps tool to scan containers for security vulnerabilities",
    "CRM for real estate agents to manage property listings and client relationships",
    "Inventory management system for small ecommerce businesses"
]

results = []

for idea in TEST_IDEAS:
    console.print(f"\n[bold cyan]Testing:[/bold cyan] {idea[:70]}...\n")

    state = create_new_state(idea)
    state = orchestrator.run(state)
    save_state(state)

    queries = state.research_plan.queries
    by_category = {}
    for q in queries:
        by_category[q.category] = by_category.get(q.category, 0) + 1

    results.append({
        "idea": idea[:50] + "...",
        "domain": state.metadata.domain,
        "total_queries": len(queries),
        "categories": by_category
    })

    console.print(f"[green]Domain:[/green] {state.metadata.domain}")
    console.print(f"[green]Queries:[/green] {len(queries)}")
    console.print(f"[green]Distribution:[/green] {by_category}\n")

# Summary table
console.print("\n[bold green]Summary Across Domains[/bold green]\n")

table = Table()
table.add_column("Idea", style="cyan", width=50)
table.add_column("Domain", style="yellow", width=12)
table.add_column("Total", style="white", width=6)
table.add_column("Competitor", style="blue", width=10)
table.add_column("Pain Points", style="red", width=12)
table.add_column("Workflow", style="green", width=10)
table.add_column("Compliance", style="magenta", width=11)

for r in results:
    table.add_row(
        r["idea"],
        r["domain"],
        str(r["total_queries"]),
        str(r["categories"].get("competitor", 0)),
        str(r["categories"].get("pain_points", 0)),
        str(r["categories"].get("workflow", 0)),
        str(r["categories"].get("compliance", 0))
    )

console.print(table)

# Save results
with open("data/domain_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

console.print("\n[dim]Results saved to: data/domain_test_results.json[/dim]")
