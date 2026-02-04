"""Tests for the PlannerAgent.

These tests verify that the planner agent correctly generates
domain-specific research queries based on product metadata.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

from agents.planner import (
    PlannerAgent,
    QueryGenerationResponse,
    QueryItem,
    QueryValidationError,
)
from app.state import Metadata, Query, State, create_new_state


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
            prompt_tokens=100,
            completion_tokens=500,
            total_tokens=600
        )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def planner_agent(mock_openai_client):
    """Create a PlannerAgent instance with mocked client."""
    return PlannerAgent("planning", mock_openai_client)


def create_mock_response(data: dict) -> MockOpenAIResponse:
    """Create a mock OpenAI response with the given data.

    Args:
        data: Dictionary to serialize as JSON response

    Returns:
        MockOpenAIResponse instance
    """
    return MockOpenAIResponse(json.dumps(data))


def create_state_with_metadata(
    raw_idea: str,
    domain: str,
    industry_tags: list,
    target_user: str,
    geography: str = "global",
    compliance_contexts: list = None
) -> State:
    """Create a state with pre-populated metadata.

    Args:
        raw_idea: The product idea
        domain: Product domain
        industry_tags: List of industry tags
        target_user: Target user description
        geography: Geographic focus
        compliance_contexts: List of compliance requirements

    Returns:
        State with metadata populated
    """
    state = create_new_state(raw_idea)
    state.metadata.domain = domain
    state.metadata.industry_tags = industry_tags
    state.metadata.target_user = target_user
    state.metadata.geography = geography
    state.metadata.compliance_contexts = compliance_contexts or []
    state.metadata.clarification_status = "confirmed"
    return state


def create_fintech_queries() -> dict:
    """Create mock query response for fintech domain."""
    return {
        "queries": [
            {"text": "freelance invoice management software comparison 2024", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "reviews"]},
            {"text": "FreshBooks vs QuickBooks freelancer pricing", "category": "competitor", "priority": "high", "expected_sources": ["pricing_pages", "comparison_sites"]},
            {"text": "best invoicing tools for freelance designers", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "articles"]},
            {"text": "Wave vs Zoho Invoice features comparison", "category": "competitor", "priority": "medium", "expected_sources": ["comparison_sites"]},
            {"text": "free invoice software for freelancers", "category": "competitor", "priority": "medium", "expected_sources": ["articles", "reviews"]},
            {"text": "freelance designers invoice tracking pain points reddit", "category": "pain_points", "priority": "high", "expected_sources": ["forums"]},
            {"text": "freelancers tax compliance challenges", "category": "pain_points", "priority": "high", "expected_sources": ["forums", "articles"]},
            {"text": "FreshBooks user complaints", "category": "pain_points", "priority": "medium", "expected_sources": ["reviews", "forums"]},
            {"text": "freelance billing workflow problems", "category": "pain_points", "priority": "medium", "expected_sources": ["forums", "articles"]},
            {"text": "QuickBooks Self-Employed negative reviews", "category": "pain_points", "priority": "medium", "expected_sources": ["reviews"]},
            {"text": "how freelance designers track expenses", "category": "workflow", "priority": "medium", "expected_sources": ["articles", "forums"]},
            {"text": "freelance invoicing process best practices", "category": "workflow", "priority": "medium", "expected_sources": ["articles"]},
            {"text": "freelancer accounting workflow", "category": "workflow", "priority": "low", "expected_sources": ["articles"]},
            {"text": "1099 tax requirements for freelancers", "category": "compliance", "priority": "high", "expected_sources": ["docs", "articles"]},
            {"text": "freelance business expense tracking tax deductions", "category": "compliance", "priority": "medium", "expected_sources": ["articles", "docs"]},
        ],
        "rationale": "Focused on understanding competitive landscape (FreshBooks, QuickBooks, Wave), user pain points around manual tracking and tax compliance, current workflows, and US tax requirements for freelancers."
    }


def create_healthcare_queries() -> dict:
    """Create mock query response for healthcare domain."""
    return {
        "queries": [
            {"text": "patient portal software for small clinics comparison", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "reviews"]},
            {"text": "athenahealth patient portal pricing", "category": "competitor", "priority": "high", "expected_sources": ["pricing_pages"]},
            {"text": "MyChart alternatives for small practices", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "articles"]},
            {"text": "SimplePractice vs Kareo patient engagement features", "category": "competitor", "priority": "medium", "expected_sources": ["comparison_sites"]},
            {"text": "best HIPAA-compliant patient portals 2024", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "articles"]},
            {"text": "small clinic patient portal implementation challenges reddit", "category": "pain_points", "priority": "high", "expected_sources": ["forums"]},
            {"text": "patient portal adoption barriers healthcare", "category": "pain_points", "priority": "high", "expected_sources": ["articles", "reports"]},
            {"text": "EHR integration difficulties small practices", "category": "pain_points", "priority": "medium", "expected_sources": ["forums", "articles"]},
            {"text": "patient portal user experience complaints", "category": "pain_points", "priority": "medium", "expected_sources": ["reviews", "forums"]},
            {"text": "telehealth setup challenges for small clinics", "category": "pain_points", "priority": "medium", "expected_sources": ["articles", "forums"]},
            {"text": "how small medical practices handle patient communications", "category": "workflow", "priority": "medium", "expected_sources": ["articles"]},
            {"text": "patient appointment scheduling workflow clinics", "category": "workflow", "priority": "medium", "expected_sources": ["articles"]},
            {"text": "medical office patient check-in process", "category": "workflow", "priority": "low", "expected_sources": ["articles"]},
            {"text": "HIPAA patient portal requirements 2024", "category": "compliance", "priority": "high", "expected_sources": ["docs", "articles"]},
            {"text": "patient data security requirements healthcare", "category": "compliance", "priority": "high", "expected_sources": ["docs"]},
            {"text": "state medical board telehealth regulations", "category": "compliance", "priority": "medium", "expected_sources": ["docs"]},
        ],
        "rationale": "Prioritized finding EMR-integrated patient portals (athenahealth, Epic MyChart), understanding adoption challenges, HIPAA compliance requirements, and small clinic workflows."
    }


def create_devtools_queries() -> dict:
    """Create mock query response for devtools domain."""
    return {
        "queries": [
            {"text": "code security analysis tools comparison 2024", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "articles"]},
            {"text": "Snyk vs SonarQube pricing features", "category": "competitor", "priority": "high", "expected_sources": ["pricing_pages", "comparison_sites"]},
            {"text": "best SAST tools for developers", "category": "competitor", "priority": "high", "expected_sources": ["comparison_sites", "reviews"]},
            {"text": "Checkmarx alternatives", "category": "competitor", "priority": "medium", "expected_sources": ["comparison_sites"]},
            {"text": "GitHub Advanced Security vs Veracode", "category": "competitor", "priority": "medium", "expected_sources": ["comparison_sites", "articles"]},
            {"text": "developer security tool frustrations reddit", "category": "pain_points", "priority": "high", "expected_sources": ["forums"]},
            {"text": "false positives security scanning tools", "category": "pain_points", "priority": "high", "expected_sources": ["forums", "articles"]},
            {"text": "security tool CI/CD integration challenges", "category": "pain_points", "priority": "medium", "expected_sources": ["forums", "articles"]},
            {"text": "vulnerability management workflow pain points", "category": "pain_points", "priority": "medium", "expected_sources": ["forums"]},
            {"text": "Snyk user complaints", "category": "pain_points", "priority": "low", "expected_sources": ["reviews"]},
            {"text": "how security teams prioritize vulnerabilities", "category": "workflow", "priority": "medium", "expected_sources": ["articles"]},
            {"text": "developer security workflow best practices", "category": "workflow", "priority": "medium", "expected_sources": ["articles"]},
            {"text": "security scanning in CI/CD pipeline", "category": "workflow", "priority": "low", "expected_sources": ["articles", "docs"]},
            {"text": "SOC2 security tool requirements", "category": "compliance", "priority": "high", "expected_sources": ["docs", "articles"]},
            {"text": "data privacy requirements security tools", "category": "compliance", "priority": "medium", "expected_sources": ["docs"]},
        ],
        "rationale": "Focused on understanding SAST/DAST landscape (Snyk, SonarQube), developer pain points with false positives and CI/CD integration, security workflows, and SOC2 requirements."
    }


class TestFintechQueries:
    """Tests for freelance invoice tool queries."""

    def test_fintech_queries(self, planner_agent, mock_openai_client):
        """Test generation of queries for fintech domain."""
        # Setup
        state = create_state_with_metadata(
            raw_idea="Build a tool for freelance designers to track invoices and expenses",
            domain="fintech",
            industry_tags=["invoicing", "freelance_tools", "expense_tracking"],
            target_user="freelance designers and creative professionals",
            geography="global",
            compliance_contexts=["tax_reporting"]
        )

        mock_data = create_fintech_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        # Execute
        result_state = planner_agent.run(state)

        # Assert
        assert len(result_state.research_plan.queries) == 15

        # Check competitor queries
        competitor_queries = [q for q in result_state.research_plan.queries if q.category == "competitor"]
        assert len(competitor_queries) == 5

        # Check pain_points queries
        pain_queries = [q for q in result_state.research_plan.queries if q.category == "pain_points"]
        assert len(pain_queries) == 5

        # Check workflow queries
        workflow_queries = [q for q in result_state.research_plan.queries if q.category == "workflow"]
        assert len(workflow_queries) == 3

        # Check compliance queries
        compliance_queries = [q for q in result_state.research_plan.queries if q.category == "compliance"]
        assert len(compliance_queries) == 2

        # Verify task board
        assert len(result_state.task_board) == 2  # Planning task + Research task
        planning_task = result_state.task_board[0]
        assert planning_task.owner == "planning"
        assert planning_task.status == "done"

        research_task = result_state.task_board[1]
        assert research_task.owner == "research"
        assert research_task.status == "pending"

    def test_fintech_competitor_names_included(self, planner_agent, mock_openai_client):
        """Test that competitor names are included in queries."""
        state = create_state_with_metadata(
            raw_idea="Build a tool for freelance designers to track invoices and expenses",
            domain="fintech",
            industry_tags=["invoicing", "freelance_tools", "expense_tracking"],
            target_user="freelance designers and creative professionals",
            geography="global",
            compliance_contexts=["tax_reporting"]
        )

        mock_data = create_fintech_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        # Check that competitor names are in queries
        all_query_text = " ".join(q.text for q in result_state.research_plan.queries)
        assert "FreshBooks" in all_query_text or "QuickBooks" in all_query_text


class TestHealthcareQueries:
    """Tests for healthcare patient portal queries."""

    def test_healthcare_queries(self, planner_agent, mock_openai_client):
        """Test generation of queries for healthcare domain."""
        state = create_state_with_metadata(
            raw_idea="HIPAA-compliant patient portal for small clinics",
            domain="healthcare",
            industry_tags=["patient_engagement", "EMR", "telehealth"],
            target_user="small medical clinics (2-10 providers)",
            geography="US",
            compliance_contexts=["HIPAA", "state_medical_boards"]
        )

        mock_data = create_healthcare_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        # Assert query count
        assert len(result_state.research_plan.queries) == 16

        # Check HIPAA queries exist
        compliance_queries = [q for q in result_state.research_plan.queries if q.category == "compliance"]
        assert len(compliance_queries) == 3

        hipaa_queries = [q for q in compliance_queries if "HIPAA" in q.text]
        assert len(hipaa_queries) >= 1

    def test_healthcare_sources_tagged(self, planner_agent, mock_openai_client):
        """Test that healthcare queries have appropriate sources."""
        state = create_state_with_metadata(
            raw_idea="HIPAA-compliant patient portal for small clinics",
            domain="healthcare",
            industry_tags=["patient_engagement", "EMR", "telehealth"],
            target_user="small medical clinics (2-10 providers)",
            geography="US",
            compliance_contexts=["HIPAA", "state_medical_boards"]
        )

        mock_data = create_healthcare_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        # Compliance queries should have 'docs' as expected source
        compliance_queries = [q for q in result_state.research_plan.queries if q.category == "compliance"]
        for query in compliance_queries:
            assert "docs" in query.expected_sources or "articles" in query.expected_sources


class TestDevtoolsQueries:
    """Tests for devtools security platform queries."""

    def test_devtools_queries(self, planner_agent, mock_openai_client):
        """Test generation of queries for devtools domain."""
        state = create_state_with_metadata(
            raw_idea="Platform to help engineers find and fix security vulnerabilities",
            domain="devtools",
            industry_tags=["security", "code_analysis", "vulnerability_management"],
            target_user="software engineers and security teams",
            geography="global",
            compliance_contexts=["SOC2", "data_privacy"]
        )

        mock_data = create_devtools_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        # Assert query count
        assert len(result_state.research_plan.queries) == 15

        # Check for security-specific queries
        all_query_text = " ".join(q.text.lower() for q in result_state.research_plan.queries)
        assert "security" in all_query_text
        assert "sast" in all_query_text or "snyk" in all_query_text.lower()

    def test_devtools_forum_queries(self, planner_agent, mock_openai_client):
        """Test that devtools queries include forum sources."""
        state = create_state_with_metadata(
            raw_idea="Platform to help engineers find and fix security vulnerabilities",
            domain="devtools",
            industry_tags=["security", "code_analysis", "vulnerability_management"],
            target_user="software engineers and security teams",
            geography="global",
            compliance_contexts=["SOC2", "data_privacy"]
        )

        mock_data = create_devtools_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        # Pain point queries should include forums
        pain_queries = [q for q in result_state.research_plan.queries if q.category == "pain_points"]
        forum_queries = [q for q in pain_queries if "forums" in q.expected_sources]
        assert len(forum_queries) >= 2


class TestQueryDistribution:
    """Tests for query distribution validation."""

    def test_query_distribution_valid(self, planner_agent, mock_openai_client):
        """Test that valid query distribution passes validation."""
        queries = [
            QueryItem(text=f"competitor query {i}", category="competitor", priority="high", expected_sources=["comparison_sites"])
            for i in range(6)
        ] + [
            QueryItem(text=f"pain point query {i}", category="pain_points", priority="high", expected_sources=["forums"])
            for i in range(6)
        ] + [
            QueryItem(text=f"workflow query {i}", category="workflow", priority="medium", expected_sources=["articles"])
            for i in range(3)
        ] + [
            QueryItem(text=f"compliance query {i}", category="compliance", priority="high", expected_sources=["docs"])
            for i in range(2)
        ]

        result = planner_agent._validate_queries(queries)
        assert result["valid"] or result["can_proceed"]

    def test_query_distribution_too_few_queries(self, planner_agent, mock_openai_client):
        """Test that too few queries fails validation."""
        queries = [
            QueryItem(text=f"query {i}", category="competitor", priority="high", expected_sources=["comparison_sites"])
            for i in range(5)
        ]

        result = planner_agent._validate_queries(queries)
        assert not result["valid"]
        assert "Too few queries" in str(result["issues"])

    def test_query_distribution_too_many_queries(self, planner_agent, mock_openai_client):
        """Test that too many queries triggers warning but can proceed."""
        queries = [
            QueryItem(text=f"competitor query {i}", category="competitor", priority="high", expected_sources=["comparison_sites"])
            for i in range(7)
        ] + [
            QueryItem(text=f"pain point query {i}", category="pain_points", priority="high", expected_sources=["forums"])
            for i in range(7)
        ] + [
            QueryItem(text=f"workflow query {i}", category="workflow", priority="medium", expected_sources=["articles"])
            for i in range(4)
        ] + [
            QueryItem(text=f"compliance query {i}", category="compliance", priority="high", expected_sources=["docs"])
            for i in range(5)
        ]

        result = planner_agent._validate_queries(queries)
        assert result["can_proceed"]  # Should allow proceeding with extra queries


class TestNoDuplicates:
    """Tests for duplicate detection."""

    def test_no_duplicates_exact_match(self, planner_agent, mock_openai_client):
        """Test detection of exact duplicate queries."""
        queries = [
            QueryItem(text="best invoicing tools for freelancers", category="competitor", priority="high", expected_sources=["comparison_sites"]),
            QueryItem(text="best invoicing tools for freelancers", category="competitor", priority="medium", expected_sources=["reviews"]),
        ]

        duplicates = planner_agent._find_duplicates(queries)
        assert len(duplicates) == 1

    def test_no_duplicates_fuzzy_match(self, planner_agent, mock_openai_client):
        """Test detection of near-duplicate queries."""
        queries = [
            QueryItem(text="best invoicing tools for freelancers", category="competitor", priority="high", expected_sources=["comparison_sites"]),
            QueryItem(text="best invoicing tool for freelancer", category="competitor", priority="medium", expected_sources=["reviews"]),
        ]

        duplicates = planner_agent._find_duplicates(queries)
        assert len(duplicates) == 1

    def test_duplicate_removal(self, planner_agent, mock_openai_client):
        """Test that duplicates are removed correctly."""
        queries = [
            QueryItem(text="best invoicing tools for freelancers", category="competitor", priority="high", expected_sources=["comparison_sites"]),
            QueryItem(text="best invoicing tools for freelancers", category="competitor", priority="medium", expected_sources=["reviews"]),
            QueryItem(text="FreshBooks pricing features", category="competitor", priority="high", expected_sources=["pricing_pages"]),
        ]

        deduplicated = planner_agent._remove_duplicates(queries)
        assert len(deduplicated) == 2


class TestPriorityAssignment:
    """Tests for priority assignment validation."""

    def test_priority_assignment_high(self, planner_agent, mock_openai_client):
        """Test that high priority is assigned to competitor pricing queries."""
        state = create_state_with_metadata(
            raw_idea="Build a tool for freelance designers to track invoices",
            domain="fintech",
            industry_tags=["invoicing", "freelance_tools"],
            target_user="freelance designers",
            geography="global",
            compliance_contexts=["tax_reporting"]
        )

        mock_data = create_fintech_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        high_priority = [q for q in result_state.research_plan.queries if q.priority == "high"]
        medium_priority = [q for q in result_state.research_plan.queries if q.priority == "medium"]
        low_priority = [q for q in result_state.research_plan.queries if q.priority == "low"]

        # Should have at least some high priority queries
        assert len(high_priority) >= 4

        # Should have distribution across priorities
        assert len(medium_priority) >= 3
        assert len(low_priority) >= 1

    def test_priority_categories(self, planner_agent, mock_openai_client):
        """Test that high priority queries are in expected categories."""
        state = create_state_with_metadata(
            raw_idea="HIPAA-compliant patient portal",
            domain="healthcare",
            industry_tags=["patient_engagement", "EMR"],
            target_user="small medical clinics",
            geography="US",
            compliance_contexts=["HIPAA"]
        )

        mock_data = create_healthcare_queries()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

        result_state = planner_agent.run(state)

        high_priority = [q for q in result_state.research_plan.queries if q.priority == "high"]

        # High priority should include competitor and compliance queries
        categories = set(q.category for q in high_priority)
        assert "competitor" in categories
        assert "compliance" in categories or "pain_points" in categories


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_metadata_domain(self, planner_agent, mock_openai_client):
        """Test that missing domain causes prerequisite check failure."""
        state = create_new_state("Some product idea")
        # Don't set domain

        result_state = planner_agent.run(state)

        # Should not have generated queries
        assert len(result_state.research_plan.queries) == 0

    def test_missing_metadata_target_user(self, planner_agent, mock_openai_client):
        """Test that missing target_user causes prerequisite check failure."""
        state = create_new_state("Some product idea")
        state.metadata.domain = "fintech"
        # Don't set target_user

        result_state = planner_agent.run(state)

        # Should not have generated queries
        assert len(result_state.research_plan.queries) == 0

    def test_already_has_research_plan(self, planner_agent, mock_openai_client):
        """Test that agent skips if research plan exists."""
        state = create_state_with_metadata(
            raw_idea="Some product idea",
            domain="fintech",
            industry_tags=["invoicing"],
            target_user="freelancers"
        )

        # Add existing queries
        state.research_plan.queries.append(
            Query(
                id="Q-existing",
                text="existing query",
                category="competitor",
                priority="high",
                expected_sources=["comparison_sites"]
            )
        )

        result_state = planner_agent.run(state)

        # Should not call LLM
        mock_openai_client.chat.completions.create.assert_not_called()

        # Should still have only the existing query
        assert len(result_state.research_plan.queries) == 1

    def test_api_error_handling(self, planner_agent, mock_openai_client):
        """Test handling of API errors with retries."""
        state = create_state_with_metadata(
            raw_idea="Test idea",
            domain="fintech",
            industry_tags=["invoicing"],
            target_user="freelancers"
        )

        from openai import OpenAIError
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("API Error")

        with pytest.raises(OpenAIError):
            planner_agent.run(state)

        # Should have attempted retries (default is 3)
        assert mock_openai_client.chat.completions.create.call_count == 3

        # Task should be marked as blocked
        assert len(state.task_board) > 0
        assert state.task_board[0].status == "blocked"

    def test_invalid_json_response(self, planner_agent, mock_openai_client):
        """Test handling of invalid JSON response."""
        state = create_state_with_metadata(
            raw_idea="Test idea",
            domain="fintech",
            industry_tags=["invoicing"],
            target_user="freelancers"
        )

        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
            "This is not valid JSON"
        )

        with pytest.raises(Exception):
            planner_agent.run(state)


class TestQueryModels:
    """Tests for Pydantic models."""

    def test_query_item_model(self):
        """Test QueryItem model validation."""
        item = QueryItem(
            text="test query",
            category="competitor",
            priority="high",
            expected_sources=["comparison_sites", "reviews"]
        )

        assert item.text == "test query"
        assert item.category == "competitor"
        assert item.priority == "high"
        assert len(item.expected_sources) == 2

    def test_query_generation_response_model(self):
        """Test QueryGenerationResponse model validation."""
        response = QueryGenerationResponse(
            queries=[
                QueryItem(
                    text="test query",
                    category="competitor",
                    priority="high",
                    expected_sources=["comparison_sites"]
                )
            ],
            rationale="Test rationale"
        )

        assert len(response.queries) == 1
        assert response.rationale == "Test rationale"

    def test_query_item_empty_sources(self):
        """Test QueryItem with empty expected sources."""
        item = QueryItem(
            text="test query",
            category="pain_points",
            priority="medium",
            expected_sources=[]
        )

        assert item.expected_sources == []


class TestStateHelperMethods:
    """Tests for state helper methods (get_queries_by_category, etc.)."""

    def test_get_queries_by_category(self):
        """Test filtering queries by category."""
        state = create_new_state("Test")
        state.research_plan.queries = [
            Query(id="1", text="comp 1", category="competitor", priority="high", expected_sources=[]),
            Query(id="2", text="comp 2", category="competitor", priority="medium", expected_sources=[]),
            Query(id="3", text="pain 1", category="pain_points", priority="high", expected_sources=[]),
        ]

        competitor_queries = state.get_queries_by_category("competitor")
        assert len(competitor_queries) == 2

        pain_queries = state.get_queries_by_category("pain_points")
        assert len(pain_queries) == 1

    def test_get_high_priority_queries(self):
        """Test filtering high priority queries."""
        state = create_new_state("Test")
        state.research_plan.queries = [
            Query(id="1", text="high 1", category="competitor", priority="high", expected_sources=[]),
            Query(id="2", text="medium 1", category="competitor", priority="medium", expected_sources=[]),
            Query(id="3", text="high 2", category="pain_points", priority="high", expected_sources=[]),
        ]

        high_priority = state.get_high_priority_queries()
        assert len(high_priority) == 2

    def test_mark_query_done(self):
        """Test marking a query as done."""
        state = create_new_state("Test")
        state.research_plan.queries = [
            Query(id="Q-123", text="test query", category="competitor", priority="high", expected_sources=[]),
        ]

        state.mark_query_done("Q-123")
        assert state.research_plan.queries[0].status == "done"

    def test_mark_query_done_not_found(self):
        """Test marking a non-existent query as done."""
        state = create_new_state("Test")
        state.research_plan.queries = [
            Query(id="Q-123", text="test query", category="competitor", priority="high", expected_sources=[]),
        ]

        # Should not raise, just do nothing
        state.mark_query_done("Q-999")
        assert state.research_plan.queries[0].status == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
