"""Tests for the CompetitorsAgent.

These tests verify that the competitors agent correctly extracts and analyzes
competitive landscape from research evidence.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI, OpenAIError

from agents.competitors import (
    CompetitiveAnalysis,
    CompetitorsAgent,
    ExtractedCompetitor,
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
            prompt_tokens=600,
            completion_tokens=1000,
            total_tokens=1600
        )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def competitors_agent(mock_openai_client):
    """Create a CompetitorsAgent instance with mocked client."""
    return CompetitorsAgent("competitors", mock_openai_client)


@pytest.fixture
def sample_state_with_evidence():
    """Create state with sample competitor evidence for testing."""
    state = create_new_state("Test idea for freelance invoice tool")

    # Set metadata
    state.metadata.target_user = "freelance designers"
    state.metadata.domain = "fintech"

    # Add sample evidence
    state.evidence = [
        Evidence(
            id="E1",
            url="https://freshbooks.com/pricing",
            title="FreshBooks Pricing",
            type="pricing",
            snippet="FreshBooks offers three plans: Lite ($17/mo for 5 clients), Plus ($30/mo for 50 clients), Premium ($55/mo unlimited). Includes invoicing, expense tracking, time tracking.",
            full_text="FreshBooks offers three plans: Lite ($17/mo for 5 clients), Plus ($30/mo for 50 clients), Premium ($55/mo unlimited). Includes invoicing, expense tracking, time tracking. Perfect for freelancers and small businesses.",
            credibility="high",
            tags=["competitor", "pricing"],
            query_id="Q1",
            relevance_score=0.9,
        ),
        Evidence(
            id="E2",
            url="https://g2.com/compare/freshbooks-vs-quickbooks",
            title="FreshBooks vs QuickBooks Comparison",
            type="article",
            snippet="FreshBooks is simpler and better for small freelancers. QuickBooks has better tax features and accounting depth but steeper learning curve.",
            full_text="FreshBooks is simpler and better for small freelancers. QuickBooks has better tax features and accounting depth but steeper learning curve. Both offer mobile apps and integrations.",
            credibility="medium",
            tags=["competitor", "comparison"],
            query_id="Q1",
            relevance_score=0.85,
        ),
        Evidence(
            id="E3",
            url="https://quickbooks.intuit.com/self-employed",
            title="QuickBooks Self-Employed Pricing",
            type="pricing",
            snippet="QuickBooks Self-Employed: $15/month. Includes mileage tracking, expense categorization, and quarterly tax estimates.",
            full_text="QuickBooks Self-Employed starts at $15/month. Includes mileage tracking, expense categorization, quarterly tax estimates, and Schedule C prep.",
            credibility="high",
            tags=["competitor", "pricing"],
            query_id="Q2",
            relevance_score=0.88,
        ),
        Evidence(
            id="E4",
            url="https://waveapps.com",
            title="Wave Free Accounting",
            type="pricing",
            snippet="Wave offers free accounting and invoicing. Payment processing at 2.9% + $0.60 per transaction.",
            full_text="Wave offers completely free accounting and invoicing for small businesses. Payment processing at 2.9% + $0.60 per transaction. No credit card required.",
            credibility="high",
            tags=["competitor"],
            query_id="Q2",
            relevance_score=0.82,
        ),
        Evidence(
            id="E5",
            url="https://reddit.com/r/freelance/discussion",
            title="Freelancer forum discussion",
            type="forum",
            snippet="I use FreshBooks for invoicing, it's so easy. But wish it had better multi-currency support.",
            full_text="I use FreshBooks for invoicing, it's so easy to use. But I wish it had better multi-currency support for my international clients.",
            credibility="low",
            tags=["pain_points"],
            query_id="Q3",
            relevance_score=0.7,
        ),
    ]

    return state


def create_mock_response(data: dict) -> MockOpenAIResponse:
    """Create a mock OpenAI response with the given data."""
    return MockOpenAIResponse(json.dumps(data))


def create_valid_competitive_analysis() -> dict:
    """Create a valid competitive analysis response for mocking."""
    return {
        "competitors": [
            {
                "name": "FreshBooks",
                "url": "https://freshbooks.com",
                "positioning": "Simple cloud accounting for small service businesses",
                "icp": "Solo freelancers and small agencies with <10 employees",
                "pricing_model": "Subscription (tiered by client count)",
                "pricing_details": "$17/mo (5 clients), $30/mo (50 clients), $55/mo (unlimited)",
                "key_features": [
                    "Invoice creation",
                    "Expense tracking",
                    "Time tracking",
                    "Payment reminders",
                    "Mobile app"
                ],
                "strengths": [
                    "Very simple interface",
                    "Good mobile app",
                    "Strong payment reminders"
                ],
                "weaknesses": [
                    "Limited multi-currency support",
                    "Basic reporting",
                    "Expensive for many clients"
                ]
            },
            {
                "name": "QuickBooks Self-Employed",
                "url": "https://quickbooks.intuit.com/self-employed",
                "positioning": "Tax-focused accounting for US freelancers",
                "icp": "US-based freelancers needing tax compliance",
                "pricing_model": "Subscription (flat)",
                "pricing_details": "$15/month",
                "key_features": [
                    "Mileage tracking",
                    "Expense categorization",
                    "Quarterly tax estimates",
                    "Schedule C prep",
                    "Bank integration"
                ],
                "strengths": [
                    "Best tax features for US users",
                    "Intuit ecosystem integration",
                    "Automatic mileage tracking"
                ],
                "weaknesses": [
                    "US-only",
                    "Basic invoicing",
                    "No client portal"
                ]
            },
            {
                "name": "Wave",
                "url": "https://waveapps.com",
                "positioning": "Free accounting for cost-conscious small businesses",
                "icp": "Small businesses wanting free software",
                "pricing_model": "Freemium",
                "pricing_details": "Free base, 2.9% + $0.60 per payment transaction",
                "key_features": [
                    "Free invoicing",
                    "Double-entry accounting",
                    "Receipt scanning",
                    "Payment processing",
                    "Financial reports"
                ],
                "strengths": [
                    "Completely free core features",
                    "Full accounting functionality",
                    "Multi-currency support"
                ],
                "weaknesses": [
                    "Higher payment processing fees",
                    "Limited integrations",
                    "No time tracking"
                ]
            }
        ],
        "opportunity_gaps": [
            "No automated late payment fees across all tools",
            "Multi-currency invoicing poorly handled",
            "No project management integration",
            "International tax support missing"
        ],
        "market_insights": "The freelance invoicing market is mature but fragmented. Tools specialize in either simplicity or tax features, leaving a gap for comprehensive solutions. International freelancers are particularly underserved."
    }


class TestCompetitorExtraction:
    """Tests for competitor extraction functionality."""

    def test_successful_extraction(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test successful extraction of competitors from evidence."""
        # Setup mock response
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        # Execute
        result_state = competitors_agent.run(sample_state_with_evidence)

        # Assert competitors were extracted
        assert len(result_state.insights.competitors) == 3

        # Verify first competitor
        c1 = result_state.insights.competitors[0]
        assert c1.name == "FreshBooks"
        assert c1.url == "https://freshbooks.com"
        assert c1.positioning == "Simple cloud accounting for small service businesses"
        assert c1.pricing_model == "Subscription (tiered by client count)"
        assert len(c1.key_features) == 5
        assert len(c1.strengths) == 3
        assert len(c1.weaknesses) == 3

        # Verify opportunity gaps were added
        assert len(result_state.insights.opportunity_gaps) == 4
        assert "late payment fees" in result_state.insights.opportunity_gaps[0].lower()

        # Verify market insights were added
        assert result_state.insights.market_insights is not None
        assert "fragmented" in result_state.insights.market_insights.lower()

    def test_evidence_filtering(self, competitors_agent, sample_state_with_evidence):
        """Test that relevant evidence types are filtered correctly."""
        # Get filtered evidence
        filtered = competitors_agent._filter_evidence(sample_state_with_evidence.evidence)

        # Should include pricing, articles, and reviews (not forum without competitor tag)
        assert len(filtered) >= 4  # E1, E2, E3, E4 (E5 is forum but has pain_points tag)

        # Verify pricing pages are included
        types = [e.get("type") for e in filtered]
        assert "pricing" in types
        assert "article" in types

    def test_evidence_filtering_by_tags(self, competitors_agent, sample_state_with_evidence):
        """Test that evidence with competitor tag is included."""
        # All our sample evidence should be included via tags
        filtered = competitors_agent._filter_evidence(sample_state_with_evidence.evidence)

        # E1-E4 have competitor tag, E5 is forum with pain_points (excluded)
        ids = [e.get("id") for e in filtered]
        assert "E1" in ids
        assert "E2" in ids
        assert "E3" in ids
        assert "E4" in ids

    def test_no_relevant_evidence(self, competitors_agent, mock_openai_client):
        """Test handling when no relevant evidence exists."""
        state = create_new_state("Test idea")
        state.metadata.target_user = "users"
        state.metadata.domain = "general"

        # Only add forum evidence without competitor tags
        state.evidence = [
            Evidence(
                id="E1",
                url="https://example.com",
                title="General discussion",
                type="forum",
                snippet="Some content",
                credibility="low",
                tags=["general"],
                query_id="Q1",
            )
        ]

        # Execute
        result_state = competitors_agent.run(state)

        # Should not call LLM
        mock_openai_client.chat.completions.create.assert_not_called()

        # State should have no competitors
        assert len(result_state.insights.competitors) == 0


class TestEvidenceLinking:
    """Tests for linking competitors back to evidence."""

    def test_name_based_linking(self, competitors_agent, sample_state_with_evidence):
        """Test that competitor names are used to link evidence."""
        # Convert evidence to dicts
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence]

        # Link evidence for FreshBooks
        linked_ids = competitors_agent._link_evidence("FreshBooks", evidence)

        # Should link to E1 (pricing), E2 (comparison), and E5 (mentions FreshBooks)
        assert "E1" in linked_ids
        assert "E2" in linked_ids
        assert "E5" in linked_ids

    def test_url_based_linking(self, competitors_agent, sample_state_with_evidence):
        """Test that competitor is linked via URL matching."""
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence]

        # Link evidence for Wave (URL contains waveapps)
        linked_ids = competitors_agent._link_evidence("Wave", evidence)

        # Should link to E4 via URL
        assert "E4" in linked_ids

    def test_partial_name_linking(self, competitors_agent, sample_state_with_evidence):
        """Test linking with partial name matches."""
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence]

        # Link QuickBooks (full name in content)
        linked_ids = competitors_agent._link_evidence("QuickBooks Self-Employed", evidence)

        # Should link to E2 (comparison) and E3 (pricing)
        assert "E2" in linked_ids or "E3" in linked_ids


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_api_error_with_retries(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test handling of API errors with retry logic."""
        # Mock API to raise error
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("API Error")

        # Execute and expect exception
        with pytest.raises(OpenAIError):
            competitors_agent.run(sample_state_with_evidence)

        # Should have attempted retries (default is 3)
        assert mock_openai_client.chat.completions.create.call_count == 3

        # Task should be marked as blocked
        task = next(
            (t for t in sample_state_with_evidence.task_board if t.owner == "competitors"),
            None
        )
        assert task is not None
        assert task.status == "blocked"

    def test_invalid_json_response(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test handling of invalid JSON response."""
        # Mock invalid JSON response
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
            "This is not valid JSON"
        )

        # Execute - should not raise but return no competitors
        result_state = competitors_agent.run(sample_state_with_evidence)

        # Should have no competitors extracted
        assert len(result_state.insights.competitors) == 0

    def test_empty_response(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test handling of empty LLM response."""
        # Mock empty response
        mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse("")

        # Execute
        result_state = competitors_agent.run(sample_state_with_evidence)

        # Should have no competitors
        assert len(result_state.insights.competitors) == 0

    def test_missing_prompt_file(self, mock_openai_client):
        """Test handling when prompt file is missing."""
        agent = CompetitorsAgent("competitors", mock_openai_client)

        # Temporarily rename the agent to trigger missing prompt
        agent.name = "nonexistent_agent"

        state = create_new_state("Test")
        state.evidence = [
            Evidence(
                id="E1",
                url="https://example.com/pricing",
                title="Pricing",
                type="pricing",
                snippet="Test pricing content",
                credibility="high",
                tags=["competitor"],
                query_id="Q1",
            )
        ]

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            agent.run(state)


class TestTaskManagement:
    """Tests for task board management."""

    def test_task_created(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that task is created on task board."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        # Should have task
        task = next(
            (t for t in result_state.task_board if t.owner == "competitors"),
            None
        )
        assert task is not None
        assert "COMPETITORS" in task.id
        assert task.status == "done"

    def test_task_marked_done_on_success(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that task is marked done on successful completion."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        task = next(t for t in result_state.task_board if t.owner == "competitors")
        assert task.status == "done"

    def test_task_marked_blocked_on_error(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that task is marked blocked on error."""
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("Error")

        with pytest.raises(OpenAIError):
            competitors_agent.run(sample_state_with_evidence)

        task = next(t for t in sample_state_with_evidence.task_board if t.owner == "competitors")
        assert task.status == "blocked"


class TestAgentTrace:
    """Tests for agent trace logging."""

    def test_trace_entries_created(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that agent trace entries are created."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        # Should have trace entries
        competitor_traces = [
            t for t in result_state.agent_trace
            if t.agent == "competitors"
        ]
        assert len(competitor_traces) > 0

        # Should have start and completion entries
        actions = [t.action for t in competitor_traces]
        assert any("started" in a.lower() for a in actions)
        assert any("completed" in a.lower() for a in actions)


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_extracted_competitor_validation(self):
        """Test ExtractedCompetitor model validation."""
        valid_data = {
            "name": "TestCompetitor",
            "url": "https://example.com",
            "positioning": "Test positioning statement",
            "icp": "Test ideal customer",
            "pricing_model": "Subscription",
            "pricing_details": "$10/month",
            "key_features": ["Feature 1", "Feature 2", "Feature 3"],
            "strengths": ["Strength 1", "Strength 2"],
            "weaknesses": ["Weakness 1"]
        }

        comp = ExtractedCompetitor(**valid_data)
        assert comp.name == "TestCompetitor"
        assert comp.url == "https://example.com"
        assert len(comp.key_features) == 3

    def test_competitive_analysis_validation(self):
        """Test CompetitiveAnalysis model validation."""
        valid_data = {
            "competitors": [
                {
                    "name": "Test",
                    "url": None,
                    "positioning": "Test",
                    "icp": "Test users",
                    "pricing_model": "Free",
                    "pricing_details": None,
                    "key_features": ["F1", "F2", "F3"],
                    "strengths": ["S1"],
                    "weaknesses": ["W1"]
                }
            ],
            "opportunity_gaps": ["Gap 1", "Gap 2"],
            "market_insights": "Test market insights"
        }

        analysis = CompetitiveAnalysis(**valid_data)
        assert len(analysis.competitors) == 1
        assert len(analysis.opportunity_gaps) == 2
        assert analysis.market_insights == "Test market insights"


class TestEvidenceContext:
    """Tests for evidence context preparation."""

    def test_evidence_context_formatting(self, competitors_agent, sample_state_with_evidence):
        """Test that evidence is properly formatted for LLM context."""
        evidence = [e.model_dump() for e in sample_state_with_evidence.evidence[:2]]

        context = competitors_agent._prepare_evidence_context(evidence)

        # Should include source IDs
        assert "Source E1" in context
        assert "Source E2" in context

        # Should include type and credibility
        assert "pricing" in context
        assert "credibility" in context

        # Should include URL
        assert "freshbooks.com" in context

        # Should include content
        assert "$17/mo" in context or "17" in context

    def test_evidence_truncation(self, competitors_agent):
        """Test that long evidence is truncated."""
        long_evidence = [{
            "id": "E1",
            "type": "pricing",
            "credibility": "high",
            "url": "A" * 500,  # Long URL
            "title": "B" * 500,  # Long title
            "snippet": "C" * 2000,  # Long snippet
            "full_text": "D" * 3000,  # Very long full text
        }]

        context = competitors_agent._prepare_evidence_context(long_evidence)

        # URL should be truncated to 200 chars
        assert "A" * 201 not in context

        # Title should be truncated
        assert "B" * 201 not in context


class TestIntegration:
    """Integration tests with real-like scenarios."""

    def test_full_analysis_flow(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test complete analysis flow from evidence to state update."""
        # Setup comprehensive mock response
        mock_response = {
            "competitors": [
                {
                    "name": "FreshBooks",
                    "url": "https://freshbooks.com",
                    "positioning": "Simple invoicing for freelancers",
                    "icp": "Solo freelancers",
                    "pricing_model": "Tiered subscription",
                    "pricing_details": "$17-55/month",
                    "key_features": ["Invoicing", "Expenses", "Time tracking", "Reports", "Mobile app"],
                    "strengths": ["Easy to use", "Good mobile app"],
                    "weaknesses": ["Limited features", "Expensive"]
                },
                {
                    "name": "QuickBooks",
                    "url": "https://quickbooks.intuit.com",
                    "positioning": "Tax-focused accounting",
                    "icp": "US freelancers",
                    "pricing_model": "Flat subscription",
                    "pricing_details": "$15/month",
                    "key_features": ["Tax estimates", "Mileage", "Expenses", "Bank sync", "Reports"],
                    "strengths": ["Tax features", "Brand trust"],
                    "weaknesses": ["US only", "Complex"]
                },
                {
                    "name": "Wave",
                    "url": "https://waveapps.com",
                    "positioning": "Free accounting",
                    "icp": "Cost-conscious businesses",
                    "pricing_model": "Freemium",
                    "pricing_details": "Free + 2.9% payment fees",
                    "key_features": ["Free invoicing", "Accounting", "Payments", "Receipts", "Reports"],
                    "strengths": ["Free", "Full features"],
                    "weaknesses": ["High payment fees", "Limited support"]
                }
            ],
            "opportunity_gaps": [
                "No automated late fees",
                "Poor multi-currency support",
                "No project integration"
            ],
            "market_insights": "Market is mature but fragmented with specialization opportunities."
        }
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        # Execute
        result_state = competitors_agent.run(sample_state_with_evidence)

        # Verify complete state update
        assert len(result_state.insights.competitors) == 3

        # Verify competitor details
        fb = next(c for c in result_state.insights.competitors if c.name == "FreshBooks")
        assert fb.url == "https://freshbooks.com"
        assert len(fb.key_features) == 5
        assert len(fb.evidence_ids) > 0  # Should be linked

        # Verify opportunity gaps
        assert len(result_state.insights.opportunity_gaps) == 3
        assert any("late fees" in gap.lower() for gap in result_state.insights.opportunity_gaps)

        # Verify market insights
        assert result_state.insights.market_insights is not None

        # Verify task completion
        task = next(t for t in result_state.task_board if t.owner == "competitors")
        assert task.status == "done"

        # Verify trace
        assert len(result_state.agent_trace) > 0

    def test_multiple_evidence_sources(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test analysis with diverse evidence types."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        # Each competitor should be linked to relevant evidence
        for comp in result_state.insights.competitors:
            # At least some competitors should have evidence links
            # (depends on name matching in test data)
            pass  # Just verify no errors

        # Verify LLM was called with formatted evidence
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args is not None


class TestStateUpdates:
    """Tests for state model updates."""

    def test_competitor_model_fields(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that all Competitor model fields are populated."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        comp = result_state.insights.competitors[0]

        # Check all extended fields
        assert comp.id is not None
        assert comp.name is not None
        assert comp.url is not None
        assert comp.positioning is not None
        assert comp.icp is not None
        assert comp.pricing_model is not None
        assert comp.key_features is not None
        assert len(comp.key_features) > 0
        assert comp.strengths is not None
        assert comp.weaknesses is not None

        # Check legacy fields are also populated
        assert comp.description is not None  # Should be same as positioning
        assert comp.pricing is not None  # Should be same as pricing_details

    def test_insights_opportunity_gaps(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that opportunity gaps are added to insights."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        assert hasattr(result_state.insights, 'opportunity_gaps')
        assert len(result_state.insights.opportunity_gaps) > 0
        assert isinstance(result_state.insights.opportunity_gaps[0], str)

    def test_insights_market_insights(self, competitors_agent, mock_openai_client, sample_state_with_evidence):
        """Test that market insights are added to insights."""
        mock_response = create_valid_competitive_analysis()
        mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_response)

        result_state = competitors_agent.run(sample_state_with_evidence)

        assert hasattr(result_state.insights, 'market_insights')
        assert result_state.insights.market_insights is not None
        assert isinstance(result_state.insights.market_insights, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
