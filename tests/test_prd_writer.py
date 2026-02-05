"""Tests for the PRDWriterAgent.

These tests verify that the PRD writer agent correctly synthesizes research insights
into a comprehensive PRD with citations and proper template population.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI, OpenAIError

from agents.prd_writer import (
    PRDContent,
    PRDWriterAgent,
    RiskItem,
)
from app.state import (
    Competitor,
    Evidence,
    Insights,
    Metadata,
    PainPoint,
    State,
    create_new_state,
)


class MockOpenAIResponse:
    """Mock response from OpenAI API."""

    def __init__(self, content: str):
        self.choices = [
            MagicMock(
                message=MagicMock(content=content, role="assistant"),
                finish_reason="stop"
            )
        ]
        self.usage = MagicMock(
            prompt_tokens=1000,
            completion_tokens=2000,
            total_tokens=3000
        )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def prd_writer_agent(mock_openai_client):
    """Create a PRDWriterAgent instance with mocked client."""
    return PRDWriterAgent("prd_writer", mock_openai_client)


@pytest.fixture
def sample_state_with_insights():
    """Create state with sample insights for testing."""
    state = create_new_state("Test idea for freelance invoice tool")

    # Set metadata
    state.metadata.domain = "fintech"
    state.metadata.target_user = "freelance designers"
    state.metadata.geography = "global"
    state.metadata.compliance_contexts = ["GDPR", "PCI-DSS"]

    # Add sample pain points
    state.insights.pain_points = [
        PainPoint(
            id="PP1",
            cluster_name="Invoice tracking overhead",
            who="Freelancers billing 5+ clients",
            what="Spend 2-3 hours weekly tracking unpaid invoices",
            why="No automated payment detection",
            description="Manual invoice tracking is time-consuming",
            severity="critical",
            frequency="mentioned in 8 sources",
            evidence_ids=["E1", "E2"],
            example_quotes=["I waste every Monday morning chasing payments"]
        ),
        PainPoint(
            id="PP2",
            cluster_name="Late payment follow-up",
            who="Solo freelancers",
            what="Manually send payment reminders",
            why="Tools lack smart reminder automation",
            description="Payment reminders are manual",
            severity="major",
            frequency="mentioned in 5 sources",
            evidence_ids=["E2", "E3"],
            example_quotes=["I feel awkward sending reminder emails"]
        ),
    ]

    # Add sample competitors
    state.insights.competitors = [
        Competitor(
            id="C1",
            name="FreshBooks",
            positioning="Simple accounting for freelancers",
            description="Accounting software",
            key_features=["Invoicing", "Expense tracking", "Time tracking"],
            strengths=["Easy to use", "Good mobile app"],
            weaknesses=["Limited automation", "Expensive tiers"],
            pricing="$15-50/month",
            evidence_ids=["E3", "E4"]
        ),
        Competitor(
            id="C2",
            name="Wave",
            positioning="Free accounting software",
            description="Free accounting",
            key_features=["Free invoicing", "Basic accounting"],
            strengths=["Free", "Good for beginners"],
            weaknesses=["Limited features", "No mobile app"],
            pricing="Free",
            evidence_ids=["E4", "E5"]
        ),
    ]

    # Add opportunity gaps
    state.insights.opportunity_gaps = [
        "No automated late fee calculation",
        "Poor multi-currency support",
        "No smart payment prediction"
    ]

    state.insights.market_insights = "The freelance invoicing market is growing at 15% annually."

    # Add sample evidence
    state.evidence = [
        Evidence(
            id="E1",
            url="https://reddit.com/r/freelance/post1",
            title="Invoice tracking is painful",
            type="forum",
            snippet="I waste every Monday morning chasing payments",
            credibility="low",
            tags=["pain_points"],
            query_id="Q1",
            relevance_score=0.8,
        ),
        Evidence(
            id="E2",
            url="https://reddit.com/r/freelance/post2",
            title="Payment follow-up struggles",
            type="forum",
            snippet="I feel awkward sending reminder emails to clients",
            credibility="low",
            tags=["pain_points"],
            query_id="Q1",
            relevance_score=0.75,
        ),
        Evidence(
            id="E3",
            url="https://g2.com/products/freshbooks/reviews",
            title="FreshBooks Review",
            type="review",
            snippet="Good for basic invoicing but lacks automation",
            credibility="medium",
            tags=["competitor"],
            query_id="Q2",
            relevance_score=0.85,
        ),
        Evidence(
            id="E4",
            url="https://freshbooks.com/pricing",
            title="FreshBooks Pricing",
            type="pricing",
            snippet="Plans from $15 to $50 per month",
            credibility="high",
            tags=["competitor", "pricing"],
            query_id="Q3",
            relevance_score=0.9,
        ),
        Evidence(
            id="E5",
            url="https://wave.com",
            title="Wave Accounting",
            type="pricing",
            snippet="Free accounting software for small businesses",
            credibility="high",
            tags=["competitor"],
            query_id="Q3",
            relevance_score=0.85,
        ),
    ]

    return state


def create_mock_response(data: dict) -> MockOpenAIResponse:
    """Create a mock OpenAI response with the given data."""
    return MockOpenAIResponse(json.dumps(data))


def create_valid_prd_response() -> dict:
    """Create a valid PRD response for mocking."""
    return {
        "product_name": "InvoiceFlow",
        "problem_statement": "Freelance designers billing 5+ clients monthly spend 2-3 hours weekly on manual invoice tracking. This administrative overhead reduces billable hours and creates cash flow uncertainty.",
        "target_users": "Freelance designers and creative professionals who bill multiple clients monthly and struggle with payment tracking and follow-up.",
        "jtbd": "When I finish a project, I want to quickly send a professional invoice, so I can get paid faster. When an invoice is overdue, I want automatic reminders sent, so I don't have to awkwardly chase payments.",
        "current_workflow": "Freelancers currently create invoices in tools like FreshBooks or Excel, manually track payment status in spreadsheets, set calendar reminders for follow-ups, and spend time each week reconciling payments.",
        "solution_overview": "InvoiceFlow automates the entire invoice lifecycle from creation to payment tracking. It syncs with bank accounts to auto-detect payments, sends smart payment reminders, and provides at-a-glance dashboards.",
        "value_proposition": "Reduce invoice management from 3 hours/week to 30 minutes with automated payment tracking and smart reminders.",
        "differentiators": [
            "Automated bank sync for payment detection",
            "AI-powered payment prediction",
            "Smart escalating reminders"
        ],
        "mvp_features": [
            "Automated payment tracking - Sync with banks to detect paid invoices",
            "Smart reminders - Auto-send payment reminders based on due dates",
            "Overdue dashboard - See all overdue invoices at a glance",
            "Professional templates - Send branded invoices quickly",
            "Client portal - Let clients view and pay invoices online"
        ],
        "phase2_features": [
            "Multi-currency support - Invoice in any currency",
            "Late fee automation - Auto-calculate and add late fees",
            "Recurring invoices - Set up automatic recurring billing"
        ],
        "future_features": [
            "Payment prediction - AI predicts when clients will pay",
            "Cash flow forecasting - Project future cash flow",
            "Expense tracking - Track business expenses"
        ],
        "non_goals": [
            "Full accounting software - We focus on invoicing only",
            "Enterprise features - Targeting freelancers, not large teams",
            "Tax filing - Integration with tax software instead"
        ],
        "user_workflows": "User creates an invoice, sends it to client, and the system tracks payment. When payment is detected, status updates automatically. If overdue, smart reminders are sent.",
        "data_integrations": "Bank account sync via Plaid API, email integration for reminders, payment processor integration for online payments.",
        "compliance_security": "GDPR compliant data handling, PCI-DSS for payment data, bank-level encryption for financial connections.",
        "success_metrics": [
            "Reduce payment tracking time from 3 hours to 30 minutes weekly",
            "Decrease days-to-payment from 45 to 25 days",
            "40% weekly active rate within 6 months",
            "90% invoice send-to-paid tracking accuracy",
            "NPS score > 50"
        ],
        "risks": [
            {
                "risk": "Bank integration complexity",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "Use established Plaid API and start with major banks only"
            },
            {
                "risk": "User adoption friction",
                "likelihood": "medium",
                "impact": "medium",
                "mitigation": "Offer import from existing tools and guided onboarding"
            },
            {
                "risk": "Competition from established players",
                "likelihood": "high",
                "impact": "medium",
                "mitigation": "Focus on automation differentiator and freelancer niche"
            }
        ],
        "rollout_plan": "Launch beta with 100 freelance designers recruited from design communities. Gather feedback for 4 weeks, iterate on core features. Public launch with ProductHunt campaign targeting creative freelancers."
    }


class TestPRDGeneration:
    """Tests for PRD generation functionality."""

    def test_successful_generation(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test successful PRD generation from insights."""
        # Setup mock response
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        # Execute
        result_state = prd_writer_agent.run(sample_state_with_insights)

        # Assert PRD was generated
        assert result_state.prd.notion_markdown
        assert len(result_state.prd.notion_markdown) > 1000

        # Check sections
        assert result_state.prd.sections
        assert result_state.prd.sections["product_name"] == "InvoiceFlow"
        assert result_state.prd.sections["problem_statement"]
        assert len(result_state.prd.sections["mvp_features"]) == 5

        # Check citation map
        assert result_state.prd.citation_map
        assert isinstance(result_state.prd.citation_map, dict)

    def test_citation_mapping(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test citation mapping builds correctly."""
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = prd_writer_agent.run(sample_state_with_insights)

        # Check citation map exists and has expected keys
        assert "pain_points" in result_state.prd.citation_map
        assert "competitors" in result_state.prd.citation_map
        assert "problem_statement" in result_state.prd.citation_map

        # Pain points should link to evidence
        assert len(result_state.prd.citation_map["pain_points"]) > 0

    def test_template_population(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test that template is properly populated."""
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = prd_writer_agent.run(sample_state_with_insights)

        prd_markdown = result_state.prd.notion_markdown

        # Check key sections are populated
        assert "InvoiceFlow" in prd_markdown
        assert "Problem Statement" in prd_markdown
        assert "Target Users" in prd_markdown
        assert "MVP Scope" in prd_markdown
        assert "Risks" in prd_markdown

        # Check tables are present
        assert "|" in prd_markdown  # Markdown table indicator

        # Check placeholders are replaced (no {{...}} remaining)
        assert "{{" not in prd_markdown


class TestInputValidation:
    """Tests for input validation."""

    def test_missing_pain_points_and_competitors(self, prd_writer_agent, mock_openai_client):
        """Test handling when no insights exist."""
        state = create_new_state("Test idea")
        state.metadata.domain = "fintech"
        state.metadata.target_user = "users"

        # No pain points or competitors
        result_state = prd_writer_agent.run(state)

        # Should return without generating PRD
        assert not result_state.prd.notion_markdown

    def test_missing_metadata(self, prd_writer_agent, mock_openai_client):
        """Test handling when metadata is missing."""
        state = create_new_state("Test idea")
        # Don't set domain or target_user

        # Add some pain points
        state.insights.pain_points = [
            PainPoint(
                id="PP1",
                description="Test pain point",
                severity="major",
            )
        ]

        result_state = prd_writer_agent.run(state)

        # Should return without generating PRD
        assert not result_state.prd.notion_markdown


class TestTableGeneration:
    """Tests for markdown table generation."""

    def test_pain_points_table(self, prd_writer_agent, sample_state_with_insights):
        """Test pain points table generation."""
        table = prd_writer_agent._build_pain_points_table(sample_state_with_insights)

        # Check table structure
        assert "| Pain Point |" in table
        assert "Invoice tracking" in table
        assert "critical" in table or "major" in table

    def test_competitors_table(self, prd_writer_agent, sample_state_with_insights):
        """Test competitors table generation."""
        table = prd_writer_agent._build_competitors_table(sample_state_with_insights)

        # Check table structure
        assert "| Competitor |" in table
        assert "FreshBooks" in table
        assert "Wave" in table

    def test_risks_table(self, prd_writer_agent):
        """Test risks table generation."""
        risks = [
            RiskItem(
                risk="Technical complexity",
                likelihood="medium",
                impact="high",
                mitigation="Start simple"
            )
        ]

        table = prd_writer_agent._build_risks_table(risks)

        # Check table structure
        assert "| Risk |" in table
        assert "Technical complexity" in table
        assert "medium" in table

    def test_empty_pain_points_table(self, prd_writer_agent):
        """Test table with no pain points."""
        state = create_new_state("Test")
        table = prd_writer_agent._build_pain_points_table(state)

        assert "No pain points" in table or table.strip() == ""


class TestContextPreparation:
    """Tests for LLM context preparation."""

    def test_context_includes_all_fields(self, prd_writer_agent, sample_state_with_insights):
        """Test that context includes all required fields."""
        context = prd_writer_agent._prepare_context(sample_state_with_insights)

        assert "domain" in context
        assert "target_user" in context
        assert "geography" in context
        assert "pain_points" in context
        assert "competitors" in context
        assert "opportunity_gaps" in context

    def test_context_formats_pain_points(self, prd_writer_agent, sample_state_with_insights):
        """Test pain points are properly formatted in context."""
        context = prd_writer_agent._prepare_context(sample_state_with_insights)

        pain_points = context["pain_points"]
        assert "Invoice tracking" in pain_points
        assert "critical" in pain_points or "major" in pain_points

    def test_context_formats_competitors(self, prd_writer_agent, sample_state_with_insights):
        """Test competitors are properly formatted in context."""
        context = prd_writer_agent._prepare_context(sample_state_with_insights)

        competitors = context["competitors"]
        assert "FreshBooks" in competitors
        assert "Wave" in competitors


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_api_error_with_retries(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test handling of API errors with retry logic."""
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("API Error")

        with pytest.raises(OpenAIError):
            prd_writer_agent.run(sample_state_with_insights)

        # Should have attempted retries
        assert mock_openai_client.chat.completions.create.call_count == 3

        # Task should be marked as blocked
        task = next(
            (t for t in sample_state_with_insights.task_board if t.owner == "prd_writer"),
            None
        )
        assert task is not None
        assert task.status == "blocked"

    def test_invalid_json_response(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test handling of invalid JSON response."""
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
            "This is not valid JSON"
        )

        result_state = prd_writer_agent.run(sample_state_with_insights)

        # Should not have PRD generated
        assert not result_state.prd.notion_markdown

    def test_empty_response(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test handling of empty LLM response."""
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse("")

        result_state = prd_writer_agent.run(sample_state_with_insights)

        # Should not have PRD generated
        assert not result_state.prd.notion_markdown


class TestTaskManagement:
    """Tests for task board management."""

    def test_task_created(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test that task is created on task board."""
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = prd_writer_agent.run(sample_state_with_insights)

        task = next(
            (t for t in result_state.task_board if t.owner == "prd_writer"),
            None
        )
        assert task is not None
        assert "PRD" in task.id
        assert task.status == "done"

    def test_task_marked_done_on_success(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test that task is marked done on successful completion."""
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = prd_writer_agent.run(sample_state_with_insights)

        task = next(t for t in result_state.task_board if t.owner == "prd_writer")
        assert task.status == "done"

    def test_task_marked_blocked_on_error(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test that task is marked blocked on error."""
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("Error")

        with pytest.raises(OpenAIError):
            prd_writer_agent.run(sample_state_with_insights)

        task = next(t for t in sample_state_with_insights.task_board if t.owner == "prd_writer")
        assert task.status == "blocked"


class TestAgentTrace:
    """Tests for agent trace logging."""

    def test_trace_entries_created(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test that agent trace entries are created."""
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = prd_writer_agent.run(sample_state_with_insights)

        prd_traces = [
            t for t in result_state.agent_trace
            if t.agent == "prd_writer"
        ]
        assert len(prd_traces) > 0

        # Should have start and completion entries
        actions = [t.action for t in prd_traces]
        assert any("started" in a.lower() for a in actions)
        assert any("completed" in a.lower() for a in actions)


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_prd_content_validation(self):
        """Test PRDContent model validation."""
        valid_data = create_valid_prd_response()
        prd = PRDContent(**valid_data)

        assert prd.product_name == "InvoiceFlow"
        assert len(prd.mvp_features) == 5
        assert len(prd.risks) == 3

    def test_risk_item_validation(self):
        """Test RiskItem model validation."""
        risk = RiskItem(
            risk="Test risk",
            likelihood="high",
            impact="medium",
            mitigation="Test mitigation"
        )

        assert risk.risk == "Test risk"
        assert risk.likelihood == "high"


class TestEvidenceSection:
    """Tests for evidence section building."""

    def test_build_evidence_section(self, prd_writer_agent, sample_state_with_insights):
        """Test evidence section building."""
        section = prd_writer_agent._build_evidence_section(
            sample_state_with_insights.insights.pain_points,
            sample_state_with_insights.evidence,
            "pain_points"
        )

        # Should include pain point names
        assert "Invoice tracking" in section

        # Should include evidence links
        assert "reddit.com" in section or "E1" in section

    def test_build_key_sources(self, prd_writer_agent, sample_state_with_insights):
        """Test key sources building."""
        citation_map = {
            "pain_points": ["E1", "E2"],
            "competitors": ["E3", "E4"]
        }

        sources = prd_writer_agent._build_key_sources(
            sample_state_with_insights.evidence,
            citation_map
        )

        # Should list sources with citations
        assert "`E" in sources  # Evidence ID markers


class TestCitations:
    """Tests for citation handling."""

    def test_add_citations_with_ids(self, prd_writer_agent):
        """Test adding citations to text."""
        text = "This is a test statement."
        evidence_ids = ["E1", "E2", "E3"]

        result = prd_writer_agent._add_citations(text, evidence_ids)

        assert "[evidence:" in result
        assert "E1" in result

    def test_add_citations_empty_list(self, prd_writer_agent):
        """Test adding citations with empty list."""
        text = "This is a test statement."
        result = prd_writer_agent._add_citations(text, [])

        assert result == text
        assert "[evidence:" not in result

    def test_add_citations_limits_to_three(self, prd_writer_agent):
        """Test that citations are limited to 3."""
        text = "Test."
        evidence_ids = ["E1", "E2", "E3", "E4", "E5"]

        result = prd_writer_agent._add_citations(text, evidence_ids)

        # Should only include first 3
        assert "E1" in result
        assert "E2" in result
        assert "E3" in result
        # E4 and E5 should not appear in citation marker
        citation_part = result.split("[evidence:")[1]
        assert "E4" not in citation_part
        assert "E5" not in citation_part


class TestIntegration:
    """Integration tests with real-like scenarios."""

    def test_full_prd_generation_flow(self, prd_writer_agent, mock_openai_client, sample_state_with_insights):
        """Test complete PRD generation flow."""
        mock_response = create_valid_prd_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = prd_writer_agent.run(sample_state_with_insights)

        # Verify complete state update
        assert result_state.prd.notion_markdown
        assert result_state.prd.sections
        assert result_state.prd.citation_map

        # Verify all major sections present
        prd = result_state.prd.notion_markdown
        assert "Overview" in prd
        assert "Problem Statement" in prd
        assert "Current State" in prd
        assert "Market Landscape" in prd
        assert "Proposed Solution" in prd
        assert "Features & Scope" in prd
        assert "Success Metrics" in prd
        assert "Risks" in prd

        # Verify task completion
        task = next(t for t in result_state.task_board if t.owner == "prd_writer")
        assert task.status == "done"

        # Verify trace
        assert len(result_state.agent_trace) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
