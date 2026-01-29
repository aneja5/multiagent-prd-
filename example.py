"""Example script demonstrating the ClarificationAgent in action.

This script shows how to use the clarification agent to extract
structured metadata from a product idea.
"""

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.clarification import ClarificationAgent
from app.config import get_config
from app.state import create_new_state


def display_metadata(state, console):
    """Display extracted metadata in a formatted table."""
    console.print("\n[bold cyan]Extracted Metadata:[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="white")

    table.add_row("Domain", state.metadata.domain)
    table.add_row("Industry Tags", ", ".join(state.metadata.industry_tags))
    table.add_row("Target User", state.metadata.target_user)
    table.add_row("Geography", state.metadata.geography)
    table.add_row(
        "Compliance",
        ", ".join(state.metadata.compliance_contexts) if state.metadata.compliance_contexts else "None"
    )
    table.add_row("Status", state.metadata.clarification_status)

    console.print(table)

    # Show assumptions and questions from agent trace
    console.print("\n[bold cyan]Agent Activity:[/bold cyan]\n")
    for entry in state.agent_trace:
        if entry.details:
            if "assumptions" in entry.details:
                console.print("[bold]Assumptions Made:[/bold]")
                for assumption in entry.details["assumptions"]:
                    console.print(f"  • {assumption}")
                console.print()

            if "questions" in entry.details:
                console.print("[bold]Clarification Questions:[/bold]")
                for question in entry.details["questions"]:
                    console.print(f"  • {question}")
                console.print()


def main():
    """Run example clarification scenarios."""
    console = Console()

    # Get configuration
    try:
        config = get_config()
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        console.print("[yellow]Make sure you have a .env file with OPENAI_API_KEY set[/yellow]")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=config.openai_api_key)

    # Initialize agent
    agent = ClarificationAgent("clarification", client)

    # Example product ideas
    examples = [
        "Build a tool for freelance designers to track invoices and expenses",
        "HIPAA-compliant patient portal for small clinics",
        "Platform to help engineers find and fix security vulnerabilities",
        "AI-powered scheduling assistant for busy executives"
    ]

    console.print(Panel.fit(
        "[bold cyan]ClarificationAgent Example[/bold cyan]\n\n"
        "This script demonstrates how the clarification agent extracts\n"
        "structured metadata from product ideas.",
        border_style="cyan"
    ))

    for i, idea in enumerate(examples, 1):
        console.print(f"\n{'=' * 70}")
        console.print(f"[bold yellow]Example {i}:[/bold yellow]")
        console.print(f"[bold]Product Idea:[/bold] {idea}\n")

        # Create state
        state = create_new_state(idea)

        # Run agent
        try:
            state = agent.run(state)
            display_metadata(state, console)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        # Ask if user wants to continue
        if i < len(examples):
            console.print("\n[dim]Press Enter to continue to next example...[/dim]")
            input()


if __name__ == "__main__":
    main()
