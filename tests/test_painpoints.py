"""Tests for the PainPointsAgent.

These tests verify that the pain points agent correctly extracts and clusters
pain points from research evidence.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI, OpenAIError

from agents.painpoints import (
    ExtractedPainPoint,
    PainPointsAgent,
    PainPointsResponse,
)
from app.state import Evidence, Metadata, State, create_new_state


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
            prompt_tokens=500,
            completion_tokens=800,
            total_tokens=1300
        )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def painpoints_agent(mock_openai_client):
    """Create a PainPointsAgent instance with mocked client."""
    return PainPointsAgent("painpoints", mock_openai_client)


@pytest.fixture
def sample_state_with_evidence():
    """Create state with sample evidence for testing."""
    state = create_new_state("Test idea for freelance invoice tool")

    # Set metadata
    state.metadata.target_user = "freelance designers"
    state.metadata.domain = "fintech"

    # Add sample evidence
    state.evidence = [
        Evidence(
            id="E1",
            url="https://reddit.com/r/freelance/post1",
            title="Invoice tracking is painful",
            type="forum",
            snippet="I waste every Monday morning chasing down payments from last month. There's no good way to track unpaid invoices.",
            full_text="I waste every Monday morning chasing down payments from last month. There's no good way to track unpaid invoices. I've tried multiple tools but none of them really solve this problem.",
            credibility="low",
            tags=["pain_points"],
            query_id="Q1",
            relevance_score=0.8,
        ),
        Evidence(
            id="E2",
            url="https://reddit.com/r/freelance/post2",
            title="Payment follow-up hell",
            type="forum",
            snippet="I have to set calendar reminders to follow up on unpaid invoices. It takes hours every week.",
            full_text="I have to set calendar reminders to follow up on unpaid invoices. It takes hours every week. My clients are notorious for paying late.",
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
            snippet="The payment reminders feature is too basic. I still have to manually check which invoices are overdue.",
            full_text="The payment reminders feature is too basic. I still have to manually check which invoices are overdue. Would love better automation here.",
            credibility="medium",
            tags=["pain_points", "competitor"],
            query_id="Q2",
            relevance_score=0.85,
        ),
        Evidence(
            id="E4",
            url="https://techcrunch.com/article",
            title="Fintech Trends 2024",
            type="article",
            snippet="The invoicing market is growing rapidly with new players emerging.",
            full_text="The invoicing market is growing rapidly with new players emerging. Competition is fierce.",
            credibility="high",
            tags=["market_research"],
            query_id="Q3",
            relevance_score=0.6,
        ),
    ]

    return state


def create_mock_response(data: dict) -> MockOpenAIResponse:
    """Create a mock OpenAI response with the given data."""
    return MockOpenAIResponse(json.dumps(data))


def create_valid_painpoints_response() -> dict:
    """Create a valid pain points response for mocking."""
    return {
        "pain_points": [
            {
                "cluster_name": "Invoice follow-up overhead",
                "who": "Freelance designers billing multiple clients monthly",
                "what": "Spend 2-3 hours every week manually tracking which invoices are unpaid and sending follow-up reminders",
                "why": "Most tools lack automated payment tracking tied to invoice due dates",
                "severity": "high",
                "frequency": "mentioned in 3 sources",
                "example_quotes": [
                    "I waste every Monday morning chasing down payments from last month",
                    "I have to set calendar reminders to follow up on unpaid invoices",
                    "I still have to manually check which invoices are overdue"
                ]
            },
            {
                "cluster_name": "Late payment tracking",
                "who": "Freelancers without automated billing systems",
                "what": "No easy way to see which invoices are overdue at a glance",
                "why": "Invoice tools show invoice status but don't highlight overdue items prominently",
                "severity": "medium",
                "frequency": "mentioned in 2 sources",
                "example_quotes": [
                    "There's no good way to track unpaid invoices",
                    "The payment reminders feature is too basic"
                ]
            }
        ],
        "rationale": "Clustered pain points around invoice tracking and follow-up as the primary themes emerging from forum posts and reviews."
    }


class TestPainPointsExtraction:
    """Tests for pain point extraction functionality."""

    def test_successful_extraction(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test successful extraction of pain points from evidence."""
        # Setup mock response
        mock_response = create_valid_painpoints_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        # Execute
        result_state = painpoints_agent.run(sample_state_with_evidence)

        # Assert pain points were extracted
        assert len(result_state.insights.pain_points) == 2

        # Verify first pain point
        pp1 = result_state.insights.pain_points[0]
        assert pp1.cluster_name == "Invoice follow-up overhead"
        assert pp1.who == "Freelance designers billing multiple clients monthly"
        assert pp1.severity == "critical"  # high maps to critical
        assert len(pp1.example_quotes) == 3
        assert len(pp1.evidence_ids) > 0

        # Verify second pain point
        pp2 = result_state.insights.pain_points[1]
        assert pp2.cluster_name == "Late payment tracking"
        assert pp2.severity == "major"  # medium maps to major

    def test_evidence_filtering(self, painpoints_agent, sample_state_with_evidence):
        """Test that only forum and review evidence is used."""
        # Get filtered evidence
        filtered = painpoints_agent._filter_evidence(sample_state_with_evidence.evidence)

        # Should include forums and reviews, exclude articles
        assert len(filtered) == 3  # E1, E2, E3 (not E4 which is article)

        # Verify types
        types = [e.get("type") for e in filtered]
        assert "forum" in types
        assert "review" in types
        assert "article" not in types

    def test_evidence_filtering_by_tags(self, painpoints_agent, sample_state_with_evidence):
        """Test that evidence with pain_points tag is included."""
        # Add article with pain_points tag
        sample_state_with_evidence.evidence.append(
            Evidence(
                id="E5",
                url="https://example.com/article",
                title="Pain points in invoicing",
                type="article",  # Not a forum/review
                snippet="Users complain about manual work",
                credibility="high",
                tags=["pain_points"],  # But has pain_points tag
                query_id="Q4",
            )
        )

        filtered = painpoints_agent._filter_evidence(sample_state_with_evidence.evidence)

        # Should include E5 because of pain_points tag
        ids = [e.get("id") for e in filtered]
        assert "E5" in ids

    def test_no_relevant_evidence(self, painpoints_agent, mock_openai_client):
        """Test handling when no relevant evidence exists."""
        state = create_new_state("Test idea")
        state.metadata.target_user = "users"
        state.metadata.domain = "general"

        # Only add article evidence (no forums/reviews)
        state.evidence = [
            Evidence(
                id="E1",
                url="https://example.com",
                title="Article",
                type="article",
                snippet="Some content",
                credibility="high",
                tags=["market"],
                query_id="Q1",
            )
        ]

        # Execute
        result_state = painpoints_agent.run(state)

        # Should not call LLM
        mock_openai_client.chat.completions.create.assert_not_called()

        # State should be unchanged
        assert len(result_state.insights.pain_points) == 0


class TestEvidenceLinking:
    """Tests for linking pain points back to evidence."""

    def test_quote_based_linking(self, painpoints_agent, sample_state_with_evidence):
        """Test that quotes are used to link pain points to evidence."""
        # Convert evidence to dicts
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence]

        # Create pain point with quote from E1
        pp = ExtractedPainPoint(
            cluster_name="Test",
            who="users",
            what="Test pain point",
            why="Test reason",
            severity="high",
            frequency="mentioned once",
            example_quotes=["I waste every Monday morning chasing down payments"]
        )

        # Link evidence
        linked_ids = painpoints_agent._link_evidence(pp, evidence)

        # Should link to E1 (contains the quote)
        assert "E1" in linked_ids

    def test_fallback_linking(self, painpoints_agent, sample_state_with_evidence):
        """Test fallback linking when no quotes match."""
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence]

        # Create pain point with non-matching quote
        pp = ExtractedPainPoint(
            cluster_name="Test",
            who="users",
            what="Test pain point",
            why="Test reason",
            severity="high",
            frequency="mentioned once",
            example_quotes=["This quote does not exist in any evidence"]
        )

        # Link evidence
        linked_ids = painpoints_agent._link_evidence(pp, evidence)

        # Should fall back to first 3 evidence items
        assert len(linked_ids) <= 3
        assert len(linked_ids) > 0


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_api_error_with_retries(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test handling of API errors with retry logic."""
        # Mock API to raise error
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("API Error")

        # Execute and expect exception
        with pytest.raises(OpenAIError):
            painpoints_agent.run(sample_state_with_evidence)

        # Should have attempted retries (default is 3)
        assert mock_openai_client.chat.completions.create.call_count == 3

        # Task should be marked as blocked
        task = next(
            (t for t in sample_state_with_evidence.task_board if t.owner == "painpoints"),
            None
        )
        assert task is not None
        assert task.status == "blocked"

    def test_invalid_json_response(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test handling of invalid JSON response."""
        # Mock invalid JSON response
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
            "This is not valid JSON"
        )

        # Execute - should not raise but return empty pain points
        result_state = painpoints_agent.run(sample_state_with_evidence)

        # Should have no pain points extracted
        assert len(result_state.insights.pain_points) == 0

    def test_empty_response(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test handling of empty LLM response."""
        # Mock empty response
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse("")

        # Execute
        result_state = painpoints_agent.run(sample_state_with_evidence)

        # Should have no pain points
        assert len(result_state.insights.pain_points) == 0

    def test_missing_prompt_file(self, mock_openai_client):
        """Test handling when prompt file is missing."""
        agent = PainPointsAgent("painpoints", mock_openai_client)

        # Temporarily rename the agent to trigger missing prompt
        agent.name = "nonexistent_agent"

        state = create_new_state("Test")
        state.evidence = [
            Evidence(
                id="E1",
                url="https://example.com",
                title="Test",
                type="forum",
                snippet="Test content",
                credibility="low",
                tags=["pain_points"],
                query_id="Q1",
            )
        ]

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            agent.run(state)


class TestTaskManagement:
    """Tests for task board management."""

    def test_task_created(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test that task is created on task board."""
        mock_response = create_valid_painpoints_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = painpoints_agent.run(sample_state_with_evidence)

        # Should have task
        task = next(
            (t for t in result_state.task_board if t.owner == "painpoints"),
            None
        )
        assert task is not None
        assert "PAINPOINTS" in task.id
        assert task.status == "done"

    def test_task_marked_done_on_success(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test that task is marked done on successful completion."""
        mock_response = create_valid_painpoints_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = painpoints_agent.run(sample_state_with_evidence)

        task = next(t for t in result_state.task_board if t.owner == "painpoints")
        assert task.status == "done"

    def test_task_marked_blocked_on_error(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test that task is marked blocked on error."""
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("Error")

        with pytest.raises(OpenAIError):
            painpoints_agent.run(sample_state_with_evidence)

        task = next(t for t in sample_state_with_evidence.task_board if t.owner == "painpoints")
        assert task.status == "blocked"


class TestAgentTrace:
    """Tests for agent trace logging."""

    def test_trace_entries_created(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test that agent trace entries are created."""
        mock_response = create_valid_painpoints_response()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = painpoints_agent.run(sample_state_with_evidence)

        # Should have trace entries
        painpoints_traces = [
            t for t in result_state.agent_trace
            if t.agent == "painpoints"
        ]
        assert len(painpoints_traces) > 0

        # Should have start and completion entries
        actions = [t.action for t in painpoints_traces]
        assert any("started" in a.lower() for a in actions)
        assert any("completed" in a.lower() for a in actions)


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_extracted_painpoint_validation(self):
        """Test ExtractedPainPoint model validation."""
        valid_data = {
            "cluster_name": "Test cluster",
            "who": "Test users",
            "what": "Test pain point",
            "why": "Test reason",
            "severity": "high",
            "frequency": "mentioned once",
            "example_quotes": ["Quote 1", "Quote 2"]
        }

        pp = ExtractedPainPoint(**valid_data)
        assert pp.cluster_name == "Test cluster"
        assert pp.severity == "high"
        assert len(pp.example_quotes) == 2

    def test_painpoints_response_validation(self):
        """Test PainPointsResponse model validation."""
        valid_data = {
            "pain_points": [
                {
                    "cluster_name": "Test",
                    "who": "users",
                    "what": "pain",
                    "why": "reason",
                    "severity": "medium",
                    "frequency": "once",
                    "example_quotes": ["quote"]
                }
            ],
            "rationale": "Test rationale"
        }

        response = PainPointsResponse(**valid_data)
        assert len(response.pain_points) == 1
        assert response.rationale == "Test rationale"

    def test_severity_mapping(self, painpoints_agent):
        """Test severity value mapping."""
        assert painpoints_agent.SEVERITY_MAP["high"] == "critical"
        assert painpoints_agent.SEVERITY_MAP["medium"] == "major"
        assert painpoints_agent.SEVERITY_MAP["low"] == "minor"


class TestEvidenceContext:
    """Tests for evidence context preparation."""

    def test_evidence_context_formatting(self, painpoints_agent, sample_state_with_evidence):
        """Test that evidence is properly formatted for LLM context."""
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence[:2]]

        context = painpoints_agent._prepare_evidence_context(evidence)

        # Should include source IDs
        assert "Source E1" in context
        assert "Source E2" in context

        # Should include type and credibility
        assert "forum" in context
        assert "credibility" in context

        # Should include content
        assert "chasing down payments" in context

    def test_evidence_truncation(self, painpoints_agent):
        """Test that long evidence is truncated."""
        long_evidence = [{
            "id": "E1",
            "type": "forum",
            "credibility": "low",
            "title": "A" * 500,  # Long title
            "snippet": "B" * 2000,  # Long snippet
            "full_text": "C" * 3000,  # Very long full text
        }]

        context = painpoints_agent._prepare_evidence_context(long_evidence)

        # Title should be truncated to 200 chars
        assert "A" * 201 not in context

        # Content should be truncated to 1500 chars
        assert "C" * 1501 not in context


class TestIntegration:
    """Integration tests with real-like scenarios."""

    def test_full_extraction_flow(self, painpoints_agent, mock_openai_client, sample_state_with_evidence):
        """Test complete extraction flow from evidence to state update."""
        # Setup comprehensive mock response
        mock_response = {
            "pain_points": [
                {
                    "cluster_name": "Manual invoice tracking",
                    "who": "Solo freelance designers",
                    "what": "Spend 3+ hours weekly on payment tracking",
                    "why": "No automated status updates",
                    "severity": "high",
                    "frequency": "mentioned in 3 sources",
                    "example_quotes": [
                        "I waste every Monday morning chasing down payments",
                        "I have to set calendar reminders"
                    ]
                },
                {
                    "cluster_name": "Basic reminder features",
                    "who": "FreshBooks users",
                    "what": "Payment reminders lack automation",
                    "why": "Tool limitations",
                    "severity": "medium",
                    "frequency": "mentioned in 1 review",
                    "example_quotes": [
                        "The payment reminders feature is too basic"
                    ]
                },
                {
                    "cluster_name": "Overdue visibility",
                    "who": "Freelancers with multiple clients",
                    "what": "Hard to see overdue invoices at a glance",
                    "why": "Poor dashboard design",
                    "severity": "low",
                    "frequency": "mentioned once",
                    "example_quotes": [
                        "There's no good way to track unpaid invoices"
                    ]
                }
            ],
            "rationale": "Grouped by workflow impact and frequency"
        }
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        # Execute
        result_state = painpoints_agent.run(sample_state_with_evidence)

        # Verify complete state update
        assert len(result_state.insights.pain_points) == 3

        # Verify severity mapping
        severities = [pp.severity for pp in result_state.insights.pain_points]
        assert "critical" in severities  # high -> critical
        assert "major" in severities     # medium -> major
        assert "minor" in severities     # low -> minor

        # Verify evidence linking
        for pp in result_state.insights.pain_points:
            assert len(pp.evidence_ids) > 0

        # Verify task completion
        task = next(t for t in result_state.task_board if t.owner == "painpoints")
        assert task.status == "done"

        # Verify trace
        assert len(result_state.agent_trace) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
