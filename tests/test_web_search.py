"""
Test web search functionality.

Tests cover:
- Basic search functionality
- Caching behavior
- Error handling and retries
- Rate limiting
- Edge cases
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from tools.web_search import (
    WebSearchTool,
    search_web,
    WebSearchError,
    RateLimitError,
    SearchAPIError,
    CACHE_DIR,
    CACHE_TTL_HOURS,
    MAX_RETRIES,
)


# Fixtures
@pytest.fixture
def mock_env_api_key(monkeypatch):
    """Set up mock API key in environment."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-api-key-12345")


@pytest.fixture
def mock_tavily_client():
    """Create a mock Tavily client."""
    with patch("tools.web_search.TavilyClient") as mock:
        yield mock


@pytest.fixture
def sample_search_response():
    """Sample Tavily API response."""
    return {
        "results": [
            {
                "url": "https://example.com/article1",
                "title": "First Article About AI",
                "content": "This is a comprehensive article about artificial intelligence and its applications.",
                "published_date": "2024-01-15",
                "score": 0.95,
            },
            {
                "url": "https://example.com/article2",
                "title": "Second Article About Machine Learning",
                "content": "Machine learning is a subset of AI that focuses on learning from data.",
                "published_date": "2024-01-10",
                "score": 0.88,
            },
            {
                "url": "https://example.com/article3",
                "title": "Third Article",
                "content": "Another relevant article about technology trends.",
                "published_date": None,
                "score": 0.75,
            },
        ]
    }


@pytest.fixture
def clean_cache():
    """Clean up cache before and after tests."""
    # Clean before
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()

    yield

    # Clean after
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()


# Unit Tests
class TestWebSearchToolInit:
    """Tests for WebSearchTool initialization."""

    def test_init_with_api_key(self, mock_tavily_client):
        """Test initialization with explicit API key."""
        tool = WebSearchTool(api_key="explicit-key")
        assert tool.api_key == "explicit-key"
        mock_tavily_client.assert_called_once_with(api_key="explicit-key")

    def test_init_with_env_var(self, mock_env_api_key, mock_tavily_client):
        """Test initialization with environment variable."""
        tool = WebSearchTool()
        assert tool.api_key == "test-api-key-12345"

    def test_init_missing_api_key(self, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TAVILY_API_KEY not found"):
            WebSearchTool()


class TestSearch:
    """Tests for search functionality."""

    def test_basic_search(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test basic search returns properly formatted results."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()
        results = tool.search("artificial intelligence", max_results=5)

        assert len(results) == 3
        assert all("url" in r for r in results)
        assert all("title" in r for r in results)
        assert all("snippet" in r for r in results)
        assert all("score" in r for r in results)

        # Check first result
        assert results[0]["url"] == "https://example.com/article1"
        assert results[0]["title"] == "First Article About AI"
        assert results[0]["score"] == 0.95

    def test_search_with_domain_filters(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test search with domain include/exclude filters."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()
        tool.search(
            "test query",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
        )

        # Verify API was called with correct parameters
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["include_domains"] == ["example.com"]
        assert call_kwargs["exclude_domains"] == ["spam.com"]

    def test_search_depth_options(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test basic and advanced search depths."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        # Test basic
        tool.search("query", search_depth="basic", use_cache=False)
        assert mock_client.search.call_args.kwargs["search_depth"] == "basic"

        # Test advanced
        tool.search("query", search_depth="advanced", use_cache=False)
        assert mock_client.search.call_args.kwargs["search_depth"] == "advanced"

    def test_empty_query_returns_empty(self, mock_env_api_key, mock_tavily_client):
        """Test empty query returns empty list without API call."""
        mock_client = MagicMock()
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        assert tool.search("") == []
        assert tool.search("   ") == []
        mock_client.search.assert_not_called()

    def test_invalid_search_depth_defaults_to_advanced(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test invalid search_depth defaults to advanced."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()
        tool.search("query", search_depth="invalid")

        assert mock_client.search.call_args.kwargs["search_depth"] == "advanced"


class TestCaching:
    """Tests for cache functionality."""

    def test_cache_hit(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test second identical search uses cache."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        # First call
        results1 = tool.search("test query", max_results=3)
        assert mock_client.search.call_count == 1

        # Second call should use cache
        results2 = tool.search("test query", max_results=3)
        assert mock_client.search.call_count == 1  # No additional call

        assert results1 == results2

    def test_cache_miss_different_query(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test different queries don't share cache."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        tool.search("query one", max_results=3)
        tool.search("query two", max_results=3)

        assert mock_client.search.call_count == 2

    def test_cache_miss_different_params(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test different parameters create different cache entries."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        tool.search("query", max_results=5)
        tool.search("query", max_results=10)

        assert mock_client.search.call_count == 2

    def test_cache_bypass(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test use_cache=False bypasses cache."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        tool.search("query", use_cache=True)
        tool.search("query", use_cache=False)

        assert mock_client.search.call_count == 2

    def test_clear_cache(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test cache clearing."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        # Create some cache entries
        tool.search("query1")
        tool.search("query2")
        tool.search("query3")

        # Clear all
        cleared = tool.clear_cache()
        assert cleared == 3

        # Verify cache is empty
        stats = tool.get_cache_stats()
        assert stats["total_entries"] == 0

    def test_cache_stats(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test cache statistics."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        # Empty cache
        stats = tool.get_cache_stats()
        assert stats["total_entries"] == 0

        # Add entries
        tool.search("query1")
        tool.search("query2")

        stats = tool.get_cache_stats()
        assert stats["total_entries"] == 2
        assert stats["total_size_kb"] > 0
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None


class TestErrorHandling:
    """Tests for error handling and retries."""

    def test_rate_limit_retry(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test retry on rate limit error."""
        mock_client = MagicMock()
        # First two calls fail with rate limit, third succeeds
        mock_client.search.side_effect = [
            Exception("Rate limit exceeded (429)"),
            Exception("Rate limit exceeded (429)"),
            sample_search_response,
        ]
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        with patch("tools.web_search.time.sleep"):  # Speed up test
            results = tool.search("query")

        assert len(results) == 3
        assert mock_client.search.call_count == 3

    def test_rate_limit_max_retries_exceeded(self, mock_env_api_key, mock_tavily_client, clean_cache):
        """Test failure after max retries on rate limit."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Rate limit exceeded (429)")
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        with patch("tools.web_search.time.sleep"):
            with pytest.raises(RateLimitError):
                tool.search("query")

        assert mock_client.search.call_count == MAX_RETRIES + 1

    def test_connection_error_retry(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test retry on connection error."""
        mock_client = MagicMock()
        mock_client.search.side_effect = [
            ConnectionError("Network unreachable"),
            sample_search_response,
        ]
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        with patch("tools.web_search.time.sleep"):
            results = tool.search("query")

        assert len(results) == 3

    def test_auth_error_no_retry(self, mock_env_api_key, mock_tavily_client, clean_cache):
        """Test authentication errors don't retry."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("401 Unauthorized - Invalid API key")
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        with pytest.raises(SearchAPIError, match="Authentication failed"):
            tool.search("query")

        # Should not retry on auth errors
        assert mock_client.search.call_count == 1

    def test_malformed_response_handling(self, mock_env_api_key, mock_tavily_client, clean_cache):
        """Test handling of malformed API responses."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"url": "https://valid.com", "title": "Valid"},  # Missing content
                {"url": "", "title": "Missing URL"},  # Empty URL
                {"title": "No URL at all"},  # No URL
                {"url": "https://complete.com", "title": "Complete", "content": "Full result", "score": "invalid"},
            ]
        }
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()
        results = tool.search("query")

        # Should only include valid results
        assert len(results) >= 1
        assert all(r["url"] for r in results)


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_spacing(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test requests are spaced appropriately."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        tool = WebSearchTool()

        with patch("tools.web_search.time.sleep") as mock_sleep:
            tool.search("query1", use_cache=False)
            tool.search("query2", use_cache=False)

            # Second request should trigger rate limit wait
            # (time since last request < MIN_REQUEST_INTERVAL_SECONDS)
            assert mock_sleep.called


class TestConvenienceFunction:
    """Tests for the search_web convenience function."""

    def test_search_web_success(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test search_web convenience function."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        results = search_web("test query", max_results=5)

        assert len(results) == 3

    def test_search_web_no_api_key(self, monkeypatch):
        """Test search_web returns empty list when no API key."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        results = search_web("query")
        assert results == []

    def test_search_web_with_all_params(self, mock_env_api_key, mock_tavily_client, sample_search_response, clean_cache):
        """Test search_web with all parameters."""
        mock_client = MagicMock()
        mock_client.search.return_value = sample_search_response
        mock_tavily_client.return_value = mock_client

        results = search_web(
            query="test",
            max_results=5,
            search_depth="basic",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
        )

        assert len(results) == 3
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["search_depth"] == "basic"


# Integration Tests (require actual API key)
@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="No API key for integration tests")
class TestIntegration:
    """Integration tests that hit the real API."""

    def test_real_search(self):
        """Test actual search against Tavily API."""
        results = search_web("OpenAI GPT-4 capabilities", max_results=5)

        assert len(results) > 0
        assert len(results) <= 5

        for result in results:
            assert "url" in result
            assert "title" in result
            assert "snippet" in result
            assert result["url"].startswith("http")

    def test_real_search_with_domains(self):
        """Test search with domain filtering."""
        results = search_web(
            "machine learning tutorial",
            max_results=5,
            include_domains=["github.com", "medium.com"],
        )

        # Results should be from specified domains
        for result in results:
            url = result["url"].lower()
            assert "github.com" in url or "medium.com" in url

    def test_real_cache_behavior(self, clean_cache):
        """Test caching with real API."""
        query = "Python asyncio tutorial 2024"

        # First call
        start1 = time.time()
        results1 = search_web(query, max_results=3)
        time1 = time.time() - start1

        # Second call (should be cached)
        start2 = time.time()
        results2 = search_web(query, max_results=3)
        time2 = time.time() - start2

        assert results1 == results2
        assert time2 < time1  # Cached should be faster

    def test_competitor_search(self):
        """Test competitor-focused search."""
        results = search_web("Zendesk vs Freshdesk pricing comparison", max_results=5)

        assert len(results) > 0

        combined_text = " ".join([r["title"] + " " + r["snippet"] for r in results]).lower()
        assert "zendesk" in combined_text or "freshdesk" in combined_text

    def test_forum_search(self):
        """Test forum/community search."""
        results = search_web("freelance invoicing pain points reddit", max_results=5)

        assert len(results) > 0
