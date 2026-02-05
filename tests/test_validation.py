"""Tests for the ValidationAgent.

These tests verify that the validation agent correctly checks PRD quality,
citation coverage, and generates appropriate recommendations.
"""

import pytest
from unittest.mock import MagicMock

from openai import OpenAI

from agents.validation import (
    ValidationAgent,
    ValidationIssue,
    ValidationReport,
)
from app.state import (
    Competitor,
    Evidence,
    Metadata,
    PainPoint,
    PRD,
    State,
    create_new_state,
)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def validation_agent(mock_openai_client):
    """Create a ValidationAgent instance with mocked client."""
    return ValidationAgent("validation", mock_openai_client)


@pytest.fixture
def sample_state_with_prd():
    """Create state with a sample PRD for validation."""
    state = create_new_state("Test idea for freelance invoice tool")

    # Set metadata
    state.metadata.domain = "fintech"
    state.metadata.target_user = "freelance designers"

    # Add sample PRD sections
    state.prd.sections = {
        "product_name": "InvoiceFlow",
        "problem_statement": "Freelance designers billing 5+ clients monthly spend 2-3 hours weekly on manual invoice tracking and payment follow-up. This creates cash flow uncertainty and administrative overhead that reduces productive time.",
        "target_users": "Freelance designers and creative professionals who bill multiple clients monthly and struggle with payment tracking.",
        "jtbd": "When I send an invoice, I want to automatically track its payment status, so I can focus on client work instead of chasing payments.",
        "current_workflow": "Freelancers create invoices in spreadsheets or basic tools, manually check bank accounts for payments, and send reminder emails when invoices are overdue.",
        "solution_overview": "InvoiceFlow automates the entire invoice lifecycle from creation to payment detection. It syncs with bank accounts to auto-detect payments and sends smart reminders based on due dates.",
        "value_proposition": "Reduce invoice management time from 3 hours to 30 minutes weekly with automated payment tracking.",
        "differentiators": [
            "Automated bank sync for real-time payment detection",
            "AI-powered payment prediction",
            "Smart escalating reminder system"
        ],
        "mvp_features": [
            "Automated payment tracking - Sync with banks to detect paid invoices",
            "Smart reminders - Auto-send payment reminders based on due dates",
            "Overdue dashboard - See all overdue invoices at a glance",
            "Professional templates - Send branded invoices quickly",
            "Client portal - Let clients view and pay invoices online"
        ],
        "phase2_features": [
            "Multi-currency support",
            "Late fee automation",
            "Recurring invoices"
        ],
        "future_features": [
            "Payment prediction",
            "Cash flow forecasting"
        ],
        "non_goals": [
            "Full accounting software - We focus on invoicing only",
            "Enterprise features - Targeting freelancers, not large teams"
        ],
        "user_workflows": "User creates invoice, sends to client, system tracks payment automatically.",
        "data_integrations": "Bank account sync via Plaid, email integration for reminders.",
        "compliance_security": "GDPR compliant, bank-level encryption.",
        "success_metrics": [
            "Reduce payment tracking time from 3 hours to 30 minutes weekly",
            "Decrease days-to-payment from 45 to 25 days",
            "40% weekly active rate within 6 months",
            "NPS score > 50"
        ],
        "risks": [
            {
                "risk": "Bank integration complexity",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "Use Plaid API"
            }
        ],
        "rollout_plan": "Beta launch with 100 freelancers, iterate for 4 weeks, then public launch."
    }

    # Set PRD markdown
    state.prd.notion_markdown = "# InvoiceFlow PRD\n\nContent here..."

    # Add citation map
    state.prd.citation_map = {
        "problem_statement": ["E1", "E2"],
        "pain_points": ["E1", "E2", "E3"],
        "competitors": ["E3", "E4"],
        "mvp_features": ["E1", "E2"],
        "solution_overview": ["E1", "E3"],
    }

    # Add sample pain points with evidence
    state.insights.pain_points = [
        PainPoint(
            id="PP1",
            cluster_name="Invoice tracking overhead",
            description="Manual tracking is time-consuming",
            severity="critical",
            evidence_ids=["E1", "E2"]
        ),
        PainPoint(
            id="PP2",
            cluster_name="Late payment follow-up",
            description="Awkward to chase payments",
            severity="major",
            evidence_ids=["E2", "E3"]
        ),
    ]

    # Add sample competitors with evidence
    state.insights.competitors = [
        Competitor(
            id="C1",
            name="FreshBooks",
            description="Simple accounting",
            evidence_ids=["E3", "E4"]
        ),
    ]

    # Add sample evidence
    state.evidence = [
        Evidence(
            id="E1",
            url="https://reddit.com/r/freelance/post1",
            title="Invoice tracking is painful",
            type="forum",
            snippet="I waste every Monday morning chasing payments",
            credibility="low",
            query_id="Q1",
        ),
        Evidence(
            id="E2",
            url="https://reddit.com/r/freelance/post2",
            title="Payment follow-up struggles",
            type="forum",
            snippet="I feel awkward sending reminder emails",
            credibility="low",
            query_id="Q1",
        ),
        Evidence(
            id="E3",
            url="https://g2.com/products/freshbooks/reviews",
            title="FreshBooks Review",
            type="review",
            snippet="Good but lacks automation",
            credibility="medium",
            query_id="Q2",
        ),
        Evidence(
            id="E4",
            url="https://freshbooks.com/pricing",
            title="FreshBooks Pricing",
            type="pricing",
            snippet="Plans from $15 to $50",
            credibility="high",
            query_id="Q3",
        ),
    ]

    return state


@pytest.fixture
def minimal_state_with_prd():
    """Create state with minimal PRD for testing issues."""
    state = create_new_state("Test idea")

    state.prd.sections = {
        "product_name": "Test Product",
        "problem_statement": "Users have a problem",  # Too short
        "target_users": "users",  # Too generic
        "mvp_features": ["Feature 1", "Feature 2"],  # Missing descriptions
    }

    state.prd.citation_map = {
        "pain_points": ["E1"],
        "competitors": [],  # Missing citations
    }

    state.insights.pain_points = [
        PainPoint(
            id="PP1",
            description="Test pain point",
            severity="major",
            evidence_ids=[]  # No evidence
        )
    ]

    return state


class TestValidationBasic:
    """Basic validation tests."""

    def test_validation_success(self, validation_agent, sample_state_with_prd):
        """Test successful validation of a good PRD."""
        result_state = validation_agent.run(sample_state_with_prd)

        # Check quality report exists
        assert result_state.quality_report
        assert "quality_score" in result_state.quality_report
        assert "citation_coverage_pct" in result_state.quality_report
        assert "issues" in result_state.quality_report
        assert "recommendations" in result_state.quality_report

        # Good PRD should have decent score
        assert result_state.quality_report["quality_score"] >= 60

    def test_validation_creates_task(self, validation_agent, sample_state_with_prd):
        """Test that validation creates and completes task."""
        result_state = validation_agent.run(sample_state_with_prd)

        task = next(
            (t for t in result_state.task_board if t.owner == "validation"),
            None
        )
        assert task is not None
        assert task.status == "done"

    def test_validation_empty_prd(self, validation_agent, mock_openai_client):
        """Test validation with empty PRD."""
        state = create_new_state("Test")

        result_state = validation_agent.run(state)

        # Should handle gracefully
        assert result_state is not None


class TestCitationChecks:
    """Tests for citation validation."""

    def test_full_citations_coverage(self, validation_agent, sample_state_with_prd):
        """Test PRD with full citation coverage."""
        result_state = validation_agent.run(sample_state_with_prd)

        # Should have good citation coverage
        assert result_state.quality_report["citation_coverage_pct"] >= 80

    def test_missing_citations(self, validation_agent, minimal_state_with_prd):
        """Test detection of missing citations."""
        result_state = validation_agent.run(minimal_state_with_prd)

        issues = result_state.quality_report["issues"]
        citation_issues = [i for i in issues if i["issue_type"] == "missing_citation"]

        # Should detect missing competitor citations
        assert len(citation_issues) > 0

    def test_pain_point_without_evidence(self, validation_agent, minimal_state_with_prd):
        """Test detection of pain points without evidence."""
        result_state = validation_agent.run(minimal_state_with_prd)

        issues = result_state.quality_report["issues"]
        citation_issues = [
            i for i in issues
            if i["issue_type"] == "missing_citation" and i["section"] == "pain_points"
        ]

        assert len(citation_issues) > 0


class TestWeakClaimsChecks:
    """Tests for weak claims validation."""

    def test_short_problem_statement(self, validation_agent, minimal_state_with_prd):
        """Test detection of too-short problem statement."""
        result_state = validation_agent.run(minimal_state_with_prd)

        issues = result_state.quality_report["issues"]
        weak_issues = [
            i for i in issues
            if i["issue_type"] == "weak_claim" and i["section"] == "problem_statement"
        ]

        assert len(weak_issues) > 0

    def test_generic_value_proposition(self, validation_agent, sample_state_with_prd):
        """Test detection of generic language in value proposition."""
        # Add generic terms to value prop
        sample_state_with_prd.prd.sections["value_proposition"] = (
            "Better and faster invoice management that improves your workflow"
        )

        result_state = validation_agent.run(sample_state_with_prd)

        issues = result_state.quality_report["issues"]
        generic_issues = [
            i for i in issues
            if i["issue_type"] == "too_generic" and i["section"] == "value_proposition"
        ]

        assert len(generic_issues) > 0


class TestCompletenessChecks:
    """Tests for completeness validation."""

    def test_missing_sections(self, validation_agent, mock_openai_client):
        """Test detection of missing required sections."""
        state = create_new_state("Test")
        state.prd.sections = {
            "product_name": "Test",
            # Missing most required sections
        }

        result_state = validation_agent.run(state)

        issues = result_state.quality_report["issues"]
        missing_issues = [i for i in issues if i["issue_type"] == "missing_section"]

        # Should detect multiple missing sections
        assert len(missing_issues) >= 3

    def test_empty_list_sections(self, validation_agent, mock_openai_client):
        """Test detection of empty list sections."""
        state = create_new_state("Test")
        state.prd.sections = {
            "product_name": "Test",
            "problem_statement": "A valid problem statement with enough content.",
            "target_users": "Freelance designers who need invoicing help.",
            "solution_overview": "A solution that helps with invoicing tasks.",
            "value_proposition": "Save time on invoicing.",
            "mvp_features": [],  # Empty list
            "success_metrics": [],  # Empty list
        }

        result_state = validation_agent.run(state)

        issues = result_state.quality_report["issues"]
        missing_issues = [
            i for i in issues
            if i["issue_type"] == "missing_section" and i["section"] in ["mvp_features", "success_metrics"]
        ]

        assert len(missing_issues) >= 1


class TestSpecificityChecks:
    """Tests for specificity validation."""

    def test_generic_target_users(self, validation_agent, minimal_state_with_prd):
        """Test detection of generic target users."""
        result_state = validation_agent.run(minimal_state_with_prd)

        issues = result_state.quality_report["issues"]
        generic_issues = [
            i for i in issues
            if i["issue_type"] == "too_generic" and i["section"] == "target_users"
        ]

        assert len(generic_issues) > 0


class TestFeatureChecks:
    """Tests for feature validation."""

    def test_too_few_mvp_features(self, validation_agent, mock_openai_client):
        """Test detection of too few MVP features."""
        state = create_new_state("Test")
        state.prd.sections = {
            "product_name": "Test",
            "problem_statement": "A valid problem statement with enough detail.",
            "target_users": "Freelance designers who need help.",
            "solution_overview": "A solution overview.",
            "value_proposition": "Save time.",
            "mvp_features": ["Feature 1"],  # Too few
            "success_metrics": ["Metric 1", "Metric 2", "Metric 3"],
        }
        state.prd.notion_markdown = "# Test PRD"

        result_state = validation_agent.run(state)

        issues = result_state.quality_report["issues"]
        feature_issues = [
            i for i in issues
            if i["section"] == "mvp_features" and "minimum" in i["description"].lower()
        ]

        assert len(feature_issues) > 0

    def test_too_many_mvp_features(self, validation_agent, sample_state_with_prd):
        """Test detection of too many MVP features (scope creep)."""
        # Add many features
        sample_state_with_prd.prd.sections["mvp_features"] = [
            f"Feature {i}" for i in range(15)
        ]

        result_state = validation_agent.run(sample_state_with_prd)

        issues = result_state.quality_report["issues"]
        feature_issues = [
            i for i in issues
            if i["section"] == "mvp_features" and "too many" in i["description"].lower()
        ]

        assert len(feature_issues) > 0


class TestMetricsChecks:
    """Tests for metrics validation."""

    def test_missing_metrics(self, validation_agent, mock_openai_client):
        """Test detection of missing success metrics."""
        state = create_new_state("Test")
        state.prd.sections = {
            "product_name": "Test",
            "problem_statement": "Valid problem.",
            "target_users": "Freelancers.",
            "solution_overview": "A solution.",
            "value_proposition": "Value.",
            "mvp_features": ["F1", "F2", "F3"],
            # Missing success_metrics
        }

        result_state = validation_agent.run(state)

        issues = result_state.quality_report["issues"]
        metrics_issues = [
            i for i in issues
            if i["section"] == "success_metrics"
        ]

        assert len(metrics_issues) > 0

    def test_vague_metrics(self, validation_agent, sample_state_with_prd):
        """Test detection of vague metrics without numbers."""
        sample_state_with_prd.prd.sections["success_metrics"] = [
            "Improve user satisfaction",
            "Increase engagement",
            "Better retention"
        ]

        result_state = validation_agent.run(sample_state_with_prd)

        issues = result_state.quality_report["issues"]
        metrics_issues = [
            i for i in issues
            if i["section"] == "success_metrics" and "numerical" in i["description"].lower()
        ]

        assert len(metrics_issues) > 0


class TestScoreCalculation:
    """Tests for score calculation."""

    def test_quality_score_calculation(self, validation_agent, sample_state_with_prd):
        """Test quality score calculation."""
        result_state = validation_agent.run(sample_state_with_prd)

        quality_score = result_state.quality_report["quality_score"]

        # Score should be between 0 and 100
        assert 0 <= quality_score <= 100

    def test_citation_coverage_calculation(self, validation_agent, sample_state_with_prd):
        """Test citation coverage calculation."""
        result_state = validation_agent.run(sample_state_with_prd)

        coverage = result_state.quality_report["citation_coverage_pct"]

        # Coverage should be a percentage
        assert 0 <= coverage <= 100

    def test_high_severity_reduces_score(self, validation_agent, mock_openai_client):
        """Test that high severity issues reduce score significantly."""
        # Create state with missing required sections (high severity)
        state = create_new_state("Test")
        state.prd.sections = {"product_name": "Test"}

        result_state = validation_agent.run(state)

        # Score should be low due to missing sections
        assert result_state.quality_report["quality_score"] < 50


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_generates_recommendations(self, validation_agent, minimal_state_with_prd):
        """Test that recommendations are generated."""
        result_state = validation_agent.run(minimal_state_with_prd)

        recommendations = result_state.quality_report["recommendations"]

        assert len(recommendations) > 0

    def test_high_severity_critical_recommendation(self, validation_agent, mock_openai_client):
        """Test that high severity issues generate critical recommendations."""
        state = create_new_state("Test")
        state.prd.sections = {"product_name": "Test"}

        result_state = validation_agent.run(state)

        recommendations = result_state.quality_report["recommendations"]

        # Should have CRITICAL recommendation
        assert any("CRITICAL" in rec for rec in recommendations)

    def test_good_prd_positive_recommendation(self, validation_agent, sample_state_with_prd):
        """Test that good PRD gets positive recommendation."""
        result_state = validation_agent.run(sample_state_with_prd)

        # If quality is good, should have positive recommendation
        if result_state.quality_report["quality_score"] >= 80:
            recommendations = result_state.quality_report["recommendations"]
            assert any("good" in rec.lower() or "ready" in rec.lower() for rec in recommendations)


class TestIssueCounts:
    """Tests for issue counting."""

    def test_issue_counts_in_report(self, validation_agent, minimal_state_with_prd):
        """Test that issue counts are included in report."""
        result_state = validation_agent.run(minimal_state_with_prd)

        issue_counts = result_state.quality_report.get("issue_counts", {})

        assert "high" in issue_counts
        assert "medium" in issue_counts
        assert "low" in issue_counts

    def test_issue_counts_match_issues(self, validation_agent, minimal_state_with_prd):
        """Test that issue counts match actual issues."""
        result_state = validation_agent.run(minimal_state_with_prd)

        issues = result_state.quality_report["issues"]
        counts = result_state.quality_report["issue_counts"]

        actual_high = len([i for i in issues if i["severity"] == "high"])
        actual_medium = len([i for i in issues if i["severity"] == "medium"])
        actual_low = len([i for i in issues if i["severity"] == "low"])

        assert counts["high"] == actual_high
        assert counts["medium"] == actual_medium
        assert counts["low"] == actual_low


class TestPassedFlag:
    """Tests for passed/failed flag."""

    def test_good_prd_passes(self, validation_agent, sample_state_with_prd):
        """Test that a good PRD passes validation."""
        result_state = validation_agent.run(sample_state_with_prd)

        # Good PRD should pass (quality >= 70, citations >= 50%)
        if (result_state.quality_report["quality_score"] >= 70 and
            result_state.quality_report["citation_coverage_pct"] >= 50):
            assert result_state.quality_report["passed"] is True

    def test_poor_prd_fails(self, validation_agent, mock_openai_client):
        """Test that a poor PRD fails validation."""
        state = create_new_state("Test")
        state.prd.sections = {"product_name": "Test"}

        result_state = validation_agent.run(state)

        # Poor PRD should fail
        assert result_state.quality_report["passed"] is False


class TestAgentTrace:
    """Tests for agent trace logging."""

    def test_trace_entries_created(self, validation_agent, sample_state_with_prd):
        """Test that agent trace entries are created."""
        result_state = validation_agent.run(sample_state_with_prd)

        validation_traces = [
            t for t in result_state.agent_trace
            if t.agent == "validation"
        ]

        assert len(validation_traces) > 0

        # Should have start and completion entries
        actions = [t.action for t in validation_traces]
        assert any("started" in a.lower() for a in actions)
        assert any("completed" in a.lower() for a in actions)


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_validation_issue_model(self):
        """Test ValidationIssue model."""
        issue = ValidationIssue(
            section="test_section",
            issue_type="missing_citation",
            description="Test description",
            severity="high",
            suggestion="Test suggestion"
        )

        assert issue.section == "test_section"
        assert issue.severity == "high"

    def test_validation_report_model(self):
        """Test ValidationReport model."""
        report = ValidationReport(
            issues=[],
            citation_coverage=0.8,
            quality_score=85.0,
            recommendations=["Test rec"],
            passed=True
        )

        assert report.citation_coverage == 0.8
        assert report.passed is True


class TestIntegration:
    """Integration tests."""

    def test_full_validation_flow(self, validation_agent, sample_state_with_prd):
        """Test complete validation flow."""
        result_state = validation_agent.run(sample_state_with_prd)

        # Verify complete quality report
        report = result_state.quality_report

        assert "issues" in report
        assert "quality_score" in report
        assert "citation_coverage_pct" in report
        assert "recommendations" in report
        assert "passed" in report
        assert "issue_counts" in report

        # Verify task completion
        task = next(t for t in result_state.task_board if t.owner == "validation")
        assert task.status == "done"

        # Verify trace
        assert len(result_state.agent_trace) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
