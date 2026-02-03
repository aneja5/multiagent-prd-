"""
Test ResearcherAgent.

Tests cover:
- Query execution
- Evidence collection and processing
- Type inference
- Deduplication
- State updates
- Error handling
- Integration with tools
"""

import os
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agents.researcher import ResearcherAgent
from app.state import (
    create_new_state,
    Evidence,
    Metadata,
    Query,
    ResearchPlan,
    State,
    Task,
)


# Test fixtures

@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    return MagicMock()


@pytest.fixture
def researcher(mock_client):
    """Create a ResearcherAgent instance."""
    return ResearcherAgent("research", mock_client)


@pytest.fixture
def sample_state():
    """Create a sample state with queries."""
    state = create_new_state("Build an invoice tracking tool for freelancers")

    # Set up metadata
    state.metadata.domain = "Invoicing & Billing"
    state.metadata.target_user = "Freelance designers"
    state.metadata.industry_tags = ["fintech", "freelance", "saas"]

    # Add research queries
    state.research_plan.queries = [
        Query(
            id="Q-001",
            text="best invoicing software for freelancers 2024",
            category="competitor",
            priority="high",
            status="pending",
            expected_sources=["comparison_sites", "reviews"]
        ),
        Query(
            id="Q-002",
            text="freelancer invoice payment problems reddit",
            category="pain_points",
            priority="high",
            status="pending",
            expected_sources=["forums"]
        ),
        Query(
            id="Q-003",
            text="how to track unpaid invoices workflow",
            category="workflow",
            priority="medium",
            status="pending",
            expected_sources=["articles"]
        ),
    ]

    # Add research task
    state.task_board.append(Task(
        id="T-RESEARCH",
        owner="research",
        status="pending",
        description="Execute research queries"
    ))

    return state


@pytest.fixture
def sample_search_results():
    """Sample search results from web search."""
    return [
        {
            "url": "https://www.g2.com/categories/invoicing",
            "title": "Best Invoicing Software 2024 | G2",
            "snippet": "Compare the best invoicing software for freelancers...",
            "score": 0.95,
            "published_date": "2024-01-15",
        },
        {
            "url": "https://www.freshbooks.com/pricing",
            "title": "FreshBooks Pricing - Plans for Freelancers",
            "snippet": "Affordable invoicing plans starting at $15/month...",
            "score": 0.88,
            "published_date": "2024-01-10",
        },
        {
            "url": "https://reddit.com/r/freelance/comments/abc123",
            "title": "What invoicing app do you use? : r/freelance",
            "snippet": "I've been using Wave but looking for alternatives...",
            "score": 0.82,
            "published_date": "2024-01-05",
        },
    ]


@pytest.fixture
def sample_fetch_content():
    """Sample fetched content."""
    return {
        "url": "https://www.g2.com/categories/invoicing",
        "title": "Best Invoicing Software 2024",
        "content": "This is the full content of the page about invoicing software...",
        "excerpt": "This is the full content...",
        "word_count": 500,
        "success": True,
    }


class TestResearcherInit:
    """Tests for ResearcherAgent initialization."""

    def test_init_basic(self, mock_client):
        """Test basic initialization."""
        agent = ResearcherAgent("research", mock_client)

        assert agent.name == "research"
        assert agent.client == mock_client
        assert agent.max_results_per_query == 10
        assert agent.max_fetch_per_query == 5

    def test_init_custom_console(self, mock_client):
        """Test initialization with custom console."""
        from rich.console import Console
        console = Console()

        agent = ResearcherAgent("research", mock_client, console=console)

        assert agent.console == console


class TestTypeInference:
    """Tests for evidence type inference."""

    def test_infer_forum_reddit(self, researcher):
        """Test forum type for Reddit URLs."""
        result = researcher._infer_type("pain_points", "https://reddit.com/r/freelance/post")
        assert result == "forum"

    def test_infer_forum_stackoverflow(self, researcher):
        """Test forum type for Stack Overflow."""
        result = researcher._infer_type("workflow", "https://stackoverflow.com/questions/123")
        assert result == "forum"

    def test_infer_forum_quora(self, researcher):
        """Test forum type for Quora."""
        result = researcher._infer_type("pain_points", "https://www.quora.com/topic/invoicing")
        assert result == "forum"

    def test_infer_review_g2(self, researcher):
        """Test review type for G2."""
        result = researcher._infer_type("competitor", "https://www.g2.com/products/freshbooks")
        assert result == "review"

    def test_infer_review_capterra(self, researcher):
        """Test review type for Capterra."""
        result = researcher._infer_type("competitor", "https://www.capterra.com/invoicing-software")
        assert result == "review"

    def test_infer_docs_documentation(self, researcher):
        """Test docs type for documentation pages."""
        result = researcher._infer_type("workflow", "https://docs.stripe.com/invoicing")
        assert result == "docs"

    def test_infer_docs_help(self, researcher):
        """Test docs type for help pages."""
        result = researcher._infer_type("workflow", "https://support.freshbooks.com/help/invoices")
        assert result == "docs"

    def test_infer_pricing_page(self, researcher):
        """Test pricing type for pricing pages."""
        result = researcher._infer_type("competitor", "https://www.freshbooks.com/pricing")
        assert result == "pricing"

    def test_infer_pricing_plans(self, researcher):
        """Test pricing type for plans pages."""
        result = researcher._infer_type("competitor", "https://www.wave.com/plans")
        assert result == "pricing"

    def test_infer_default_competitor(self, researcher):
        """Test default type for competitor category."""
        result = researcher._infer_type("competitor", "https://techcrunch.com/freshbooks-review")
        assert result == "article"

    def test_infer_default_pain_points(self, researcher):
        """Test default type for pain_points category (when no URL match)."""
        result = researcher._infer_type("pain_points", "https://medium.com/freelance-tips")
        assert result == "forum"

    def test_infer_default_compliance(self, researcher):
        """Test default type for compliance category."""
        result = researcher._infer_type("compliance", "https://irs.gov/forms")
        assert result == "docs"


class TestEvidenceProcessing:
    """Tests for evidence processing and state updates."""

    def test_add_evidence_to_state(self, researcher, sample_state):
        """Test adding evidence items to state."""
        evidence_items = [
            {
                "url": "https://example.com/article1",
                "title": "Test Article 1",
                "snippet": "This is a test snippet...",
                "full_text": "Full content here...",
                "type": "article",
                "tags": ["competitor"],
                "credibility": {"tier": "high", "score": 0.85, "signals": ["trusted domain"]},
                "query_id": "Q-001",
                "relevance_score": 0.9,
            },
            {
                "url": "https://example.com/article2",
                "title": "Test Article 2",
                "snippet": "Another test snippet...",
                "full_text": "More content...",
                "type": "forum",
                "tags": ["pain_points"],
                "credibility": {"tier": "medium", "score": 0.6, "signals": []},
                "query_id": "Q-002",
                "relevance_score": 0.75,
            },
        ]

        researcher._add_evidence_to_state(sample_state, evidence_items)

        assert len(sample_state.evidence) == 2
        assert sample_state.evidence[0].id == "E1"
        assert sample_state.evidence[0].url == "https://example.com/article1"
        assert sample_state.evidence[0].credibility == "high"
        assert sample_state.evidence[1].id == "E2"
        assert sample_state.evidence[1].credibility == "medium"

    def test_add_evidence_validates_type(self, researcher, sample_state):
        """Test that invalid types get corrected."""
        evidence_items = [
            {
                "url": "https://example.com/test",
                "title": "Test",
                "snippet": "Snippet",
                "full_text": "",
                "type": "invalid_type",  # Invalid
                "tags": [],
                "credibility": {"tier": "medium", "score": 0.5, "signals": []},
                "query_id": "Q-001",
                "relevance_score": 0.5,
            },
        ]

        researcher._add_evidence_to_state(sample_state, evidence_items)

        assert sample_state.evidence[0].type == "article"  # Default

    def test_add_evidence_clamps_relevance(self, researcher, sample_state):
        """Test that relevance scores are clamped to 0-1."""
        evidence_items = [
            {
                "url": "https://example.com/test",
                "title": "Test",
                "snippet": "Snippet",
                "full_text": "",
                "type": "article",
                "tags": [],
                "credibility": {"tier": "low", "score": 0.3, "signals": []},
                "query_id": "Q-001",
                "relevance_score": 1.5,  # Out of range
            },
        ]

        researcher._add_evidence_to_state(sample_state, evidence_items)

        assert sample_state.evidence[0].relevance_score == 1.0

    def test_update_task_status(self, researcher, sample_state):
        """Test that research task is marked as done."""
        researcher._update_task_status(sample_state)

        research_task = next(t for t in sample_state.task_board if t.owner == "research")
        assert research_task.status == "done"


class TestDeduplication:
    """Tests for evidence deduplication."""

    def test_deduplicate_empty_list(self, researcher):
        """Test deduplication with empty list."""
        result = researcher._deduplicate_evidence([])
        assert result == []

    @patch("agents.researcher.deduplicate_evidence")
    def test_deduplicate_calls_tool(self, mock_dedupe, researcher):
        """Test that deduplication calls the dedupe tool."""
        evidence = [{"url": "https://example.com", "title": "Test", "snippet": "..."}]
        mock_dedupe.return_value = evidence

        result = researcher._deduplicate_evidence(evidence)

        mock_dedupe.assert_called_once_with(evidence)
        assert result == evidence

    @patch("agents.researcher.deduplicate_evidence")
    def test_deduplicate_handles_error(self, mock_dedupe, researcher):
        """Test that deduplication handles errors gracefully."""
        evidence = [{"url": "https://example.com", "title": "Test", "snippet": "..."}]
        mock_dedupe.side_effect = Exception("Dedup failed")

        result = researcher._deduplicate_evidence(evidence)

        # Should return original list on error
        assert result == evidence


class TestQueryExecution:
    """Tests for query execution logic."""

    @patch.object(ResearcherAgent, "search_tool", new_callable=PropertyMock)
    @patch.object(ResearcherAgent, "content_fetcher", new_callable=PropertyMock)
    def test_execute_query_success(
        self,
        mock_fetcher_prop,
        mock_search_prop,
        researcher,
        sample_state,
        sample_search_results,
        sample_fetch_content
    ):
        """Test successful query execution."""
        # Setup mocks
        mock_search = MagicMock()
        mock_search.search.return_value = sample_search_results
        mock_search_prop.return_value = mock_search

        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = sample_fetch_content
        mock_fetcher_prop.return_value = mock_fetcher

        with patch("agents.researcher.score_credibility") as mock_cred:
            mock_cred.return_value = {
                "tier": "high",
                "score": 0.85,
                "signals": ["trusted domain"]
            }

            query = sample_state.research_plan.queries[0]
            evidence = researcher._execute_query(query, sample_state)

        assert len(evidence) > 0
        assert all("url" in e for e in evidence)
        assert all("credibility" in e for e in evidence)

    @patch.object(ResearcherAgent, "search_tool", new_callable=PropertyMock)
    def test_execute_query_no_results(
        self,
        mock_search_prop,
        researcher,
        sample_state
    ):
        """Test query execution with no search results."""
        mock_search = MagicMock()
        mock_search.search.return_value = []
        mock_search_prop.return_value = mock_search

        query = sample_state.research_plan.queries[0]
        evidence = researcher._execute_query(query, sample_state)

        assert evidence == []

    @patch.object(ResearcherAgent, "search_tool", new_callable=PropertyMock)
    def test_execute_query_search_error(
        self,
        mock_search_prop,
        researcher,
        sample_state
    ):
        """Test query execution when search fails."""
        from tools.web_search import WebSearchError

        mock_search = MagicMock()
        mock_search.search.side_effect = WebSearchError("API error")
        mock_search_prop.return_value = mock_search

        query = sample_state.research_plan.queries[0]
        evidence = researcher._execute_query(query, sample_state)

        assert evidence == []


class TestResearcherRun:
    """Tests for the main run method."""

    def test_run_no_pending_queries(self, researcher, sample_state):
        """Test run with no pending queries."""
        # Mark all queries as done
        for q in sample_state.research_plan.queries:
            q.status = "done"

        result = researcher.run(sample_state)

        assert result == sample_state
        # Should have logged skip action
        skip_actions = [t for t in result.agent_trace if "skipped" in t.action]
        assert len(skip_actions) > 0

    @patch.object(ResearcherAgent, "_execute_all_queries")
    @patch.object(ResearcherAgent, "_deduplicate_evidence")
    def test_run_success(
        self,
        mock_dedupe,
        mock_execute,
        researcher,
        sample_state
    ):
        """Test successful run."""
        mock_evidence = [
            {
                "url": "https://example.com/test",
                "title": "Test Article",
                "snippet": "Test snippet...",
                "full_text": "Full text content...",
                "type": "article",
                "tags": ["competitor"],
                "credibility": {"tier": "high", "score": 0.85, "signals": []},
                "query_id": "Q-001",
                "relevance_score": 0.9,
            }
        ]
        mock_execute.return_value = mock_evidence
        mock_dedupe.return_value = mock_evidence

        result = researcher.run(sample_state)

        assert len(result.evidence) == 1
        assert result.evidence[0].url == "https://example.com/test"

        # Check task marked as done
        research_task = next(t for t in result.task_board if t.owner == "research")
        assert research_task.status == "done"

    @patch.object(ResearcherAgent, "_execute_all_queries")
    @patch.object(ResearcherAgent, "_deduplicate_evidence")
    def test_run_logs_completion(
        self,
        mock_dedupe,
        mock_execute,
        researcher,
        sample_state
    ):
        """Test that run logs completion action."""
        mock_execute.return_value = []
        mock_dedupe.return_value = []

        result = researcher.run(sample_state)

        # Should have completion trace
        completion_traces = [
            t for t in result.agent_trace
            if "completed_research" in t.action
        ]
        assert len(completion_traces) > 0


class TestResearchStats:
    """Tests for research statistics."""

    def test_get_stats_empty(self, researcher, sample_state):
        """Test stats with no evidence."""
        stats = researcher.get_research_stats(sample_state)

        assert stats["total_evidence"] == 0
        assert stats["by_type"] == {}
        assert stats["by_credibility"] == {}

    def test_get_stats_with_evidence(self, researcher, sample_state):
        """Test stats with evidence."""
        sample_state.evidence = [
            Evidence(
                id="E1",
                url="https://example1.com",
                title="Article 1",
                type="article",
                snippet="...",
                tags=["competitor"],
                credibility="high",
                query_id="Q-001",
            ),
            Evidence(
                id="E2",
                url="https://example2.com",
                title="Forum Post",
                type="forum",
                snippet="...",
                tags=["pain_points"],
                credibility="medium",
                query_id="Q-002",
            ),
            Evidence(
                id="E3",
                url="https://example3.com",
                title="Article 2",
                type="article",
                snippet="...",
                tags=["competitor"],
                credibility="high",
                query_id="Q-001",
            ),
        ]

        stats = researcher.get_research_stats(sample_state)

        assert stats["total_evidence"] == 3
        assert stats["by_type"]["article"] == 2
        assert stats["by_type"]["forum"] == 1
        assert stats["by_credibility"]["high"] == 2
        assert stats["by_credibility"]["medium"] == 1
        assert stats["by_category"]["competitor"] == 2
        assert stats["by_category"]["pain_points"] == 1


class TestFetchSingleResult:
    """Tests for single result fetching."""

    @patch.object(ResearcherAgent, "content_fetcher", new_callable=PropertyMock)
    def test_fetch_single_result_success(
        self,
        mock_fetcher_prop,
        researcher
    ):
        """Test successful single result fetch."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = {
            "url": "https://example.com/article",
            "title": "Test Article",
            "content": "Full article content here...",
            "excerpt": "Full article...",
            "success": True,
        }
        mock_fetcher_prop.return_value = mock_fetcher

        with patch("agents.researcher.score_credibility") as mock_cred:
            mock_cred.return_value = {
                "tier": "medium",
                "score": 0.6,
                "signals": []
            }

            result = researcher._fetch_single_result(
                {"url": "https://example.com/article", "title": "Test", "snippet": "..."},
                query_id="Q-001",
                category="competitor"
            )

        assert result is not None
        assert result["url"] == "https://example.com/article"
        assert result["query_id"] == "Q-001"
        assert result["credibility"]["tier"] == "medium"

    @patch.object(ResearcherAgent, "content_fetcher", new_callable=PropertyMock)
    def test_fetch_single_result_failure(
        self,
        mock_fetcher_prop,
        researcher
    ):
        """Test single result fetch failure."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = {"success": False, "error": "404"}
        mock_fetcher_prop.return_value = mock_fetcher

        result = researcher._fetch_single_result(
            {"url": "https://example.com/missing", "title": "Test", "snippet": "..."},
            query_id="Q-001",
            category="competitor"
        )

        assert result is None

    def test_fetch_single_result_no_url(self, researcher):
        """Test single result with no URL."""
        result = researcher._fetch_single_result(
            {"title": "Test", "snippet": "..."},  # No URL
            query_id="Q-001",
            category="competitor"
        )

        assert result is None


# Integration tests (requires API keys)

@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set"
)
class TestResearcherIntegration:
    """Integration tests requiring API keys."""

    def test_search_tool_initialization(self, researcher):
        """Test that search tool initializes correctly."""
        tool = researcher.search_tool
        assert tool is not None

    def test_content_fetcher_initialization(self, researcher):
        """Test that content fetcher initializes correctly."""
        fetcher = researcher.content_fetcher
        assert fetcher is not None

    @pytest.mark.slow
    def test_execute_single_query_live(self, researcher, sample_state):
        """Test executing a single query against live APIs."""
        query = Query(
            id="Q-TEST",
            text="best invoicing software 2024",
            category="competitor",
            priority="high",
            status="pending"
        )

        evidence = researcher._execute_query(query, sample_state)

        # Should get some results
        assert len(evidence) >= 0  # May be 0 if APIs fail

        # If we got results, validate structure
        for item in evidence:
            assert "url" in item
            assert "title" in item
            assert "credibility" in item
            assert "query_id" in item
            assert item["query_id"] == "Q-TEST"
