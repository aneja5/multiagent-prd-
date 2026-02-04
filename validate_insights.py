"""
Validate quality of extracted insights
"""

from app.state import load_state
from pathlib import Path
import os
from rich.console import Console
from rich.table import Table
import json

console = Console()

# Find latest run
runs_dir = Path('data/runs')
run_files = sorted(runs_dir.glob('*.json'), key=os.path.getmtime, reverse=True)

if not run_files:
    console.print("[red]No runs found[/red]")
    exit(1)

latest = run_files[0]
state = load_state(latest.stem)

console.print(f"\n[bold cyan]Validating Insights Quality[/bold cyan]")
console.print(f"Run: {latest.stem}\n")

# Validate Pain Points
console.print("[bold yellow]Pain Points Validation:[/bold yellow]\n")

pp_issues = []
for i, pp in enumerate(state.insights.pain_points, 1):
    issues = []

    # Check completeness
    if not pp.get("cluster_name"):
        issues.append("Missing cluster_name")
    if not pp.get("who"):
        issues.append("Missing who")
    if not pp.get("what"):
        issues.append("Missing what")
    if not pp.get("why"):
        issues.append("Missing why")

    # Check specificity
    if pp.get("who") and len(pp["who"]) < 20:
        issues.append("'who' too generic (too short)")
    if pp.get("who") and "user" in pp["who"].lower() and len(pp["who"]) < 30:
        issues.append("'who' uses generic 'user' term")

    if pp.get("what") and len(pp["what"]) < 30:
        issues.append("'what' too vague (too short)")

    # Check quotes
    if not pp.get("example_quotes") or len(pp["example_quotes"]) == 0:
        issues.append("No example quotes")
    elif len(pp["example_quotes"]) < 2:
        issues.append("Only 1 quote (should have 2-3)")

    # Check evidence
    if not pp.get("evidence_ids") or len(pp["evidence_ids"]) == 0:
        issues.append("No evidence linked")

    if issues:
        pp_issues.append({"id": i, "cluster": pp.get("cluster_name", "Unknown"), "issues": issues})

if pp_issues:
    console.print("[yellow]Issues found:[/yellow]")
    for item in pp_issues:
        console.print(f"  PP{item['id']}: {item['cluster']}")
        for issue in item['issues']:
            console.print(f"    - {issue}")
    console.print()
else:
    console.print("[green]✓ All pain points look good![/green]\n")

# Validate Competitors
console.print("[bold yellow]Competitors Validation:[/bold yellow]\n")

comp_issues = []
for i, comp in enumerate(state.insights.competitors, 1):
    issues = []

    # Check completeness
    if not comp.get("name"):
        issues.append("Missing name")
    if not comp.get("positioning"):
        issues.append("Missing positioning")
    if not comp.get("key_features"):
        issues.append("Missing key_features")

    # Check specificity
    if comp.get("positioning") and len(comp["positioning"]) < 30:
        issues.append("Positioning too vague")

    if comp.get("icp") and ("user" in comp["icp"].lower() or len(comp["icp"]) < 20):
        issues.append("ICP too generic")

    # Check features
    if comp.get("key_features") and len(comp["key_features"]) < 3:
        issues.append("Too few features (<3)")
    if comp.get("key_features") and len(comp["key_features"]) > 10:
        issues.append("Too many features (>10, should be top 5-7)")

    # Check strengths/weaknesses
    if not comp.get("strengths"):
        issues.append("Missing strengths")
    if not comp.get("weaknesses"):
        issues.append("Missing weaknesses")

    if issues:
        comp_issues.append({"id": i, "name": comp.get("name", "Unknown"), "issues": issues})

if comp_issues:
    console.print("[yellow]Issues found:[/yellow]")
    for item in comp_issues:
        console.print(f"  C{item['id']}: {item['name']}")
        for issue in item['issues']:
            console.print(f"    - {issue}")
    console.print()
else:
    console.print("[green]✓ All competitors look good![/green]\n")

# Summary scores
pp_score = max(0, 100 - (len(pp_issues) * 10))
comp_score = max(0, 100 - (len(comp_issues) * 10))

console.print("[bold]Quality Scores:[/bold]")
console.print(f"  Pain Points: {pp_score}/100")
console.print(f"  Competitors: {comp_score}/100\n")

if pp_score >= 80 and comp_score >= 80:
    console.print("[green]✓ High quality insights![/green]\n")
elif pp_score >= 60 and comp_score >= 60:
    console.print("[yellow]⚠ Moderate quality - some improvements needed[/yellow]\n")
else:
    console.print("[red]✗ Low quality - significant improvements needed[/red]\n")

# Save validation report
validation_report = {
    "run_id": state.run_id,
    "pain_points": {
        "count": len(state.insights.pain_points),
        "issues": pp_issues,
        "score": pp_score,
    },
    "competitors": {
        "count": len(state.insights.competitors),
        "issues": comp_issues,
        "score": comp_score,
    }
}

with open(f"data/validation_report_{state.run_id}.json", "w") as f:
    json.dump(validation_report, f, indent=2)

console.print(f"[dim]Validation report: data/validation_report_{state.run_id}.json[/dim]\n")
