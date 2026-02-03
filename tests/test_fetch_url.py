"""
Test URL fetching functionality.

Tests cover:
- Basic fetch functionality
- Caching behavior
- Error handling and retries
- Rate limiting
- Content validation
- Edge cases
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest
import requests as requests_lib

from tools.fetch_url import (
    ContentFetcher,
    fetch_url,
    fetch_urls,
    ContentFetchError,
    RateLimitError,
    NetworkError,
    ContentValidationError,
    CACHE_DIR,
    CACHE_TTL_HOURS,
    MAX_RETRIES,
)


# Patch targets - patch the global modules since they're imported
REQUESTS_GET_PATCH = "requests.get"
TIME_SLEEP_PATCH = "time.sleep"


# Fixtures
@pytest.fixture
def sample_markdown_content():
    """Sample markdown content as Jina Reader would return."""
    return """# Sample Article Title

Author: John Doe
Published: 2024-01-15

This is the introduction paragraph of the article. It provides an overview
of what the article will cover.

## Section One

This section covers the first major topic. It includes several paragraphs
of detailed information about the subject matter.

The content continues with more details and examples.

## Section Two

Another section with more information. This demonstrates how Jina Reader
formats content from web pages into clean markdown.

### Subsection

Even more detailed content here with specific examples and code snippets.

## Conclusion

The article wraps up with a summary of the key points covered.
"""


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


@pytest.fixture
def mock_response(sample_markdown_content):
    """Create a mock response object."""
    mock = Mock()
    mock.status_code = 200
    mock.text = sample_markdown_content
    return mock


# Unit Tests
class TestContentFetcherInit:
    """Tests for ContentFetcher initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        fetcher = ContentFetcher()
        assert fetcher.base_url == "https://r.jina.ai"
        assert fetcher.timeout == 30

    def test_init_custom_base_url(self):
        """Test initialization with custom base URL."""
        fetcher = ContentFetcher(base_url="https://custom.jina.ai/")
        assert fetcher.base_url == "https://custom.jina.ai"  # Trailing slash removed

    def test_init_with_api_key(self, monkeypatch):
        """Test initialization with API key from env."""
        monkeypatch.setenv("JINA_API_KEY", "test-key-123")
        fetcher = ContentFetcher()
        assert fetcher.api_key == "test-key-123"


class TestURLValidation:
    """Tests for URL validation."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        fetcher = ContentFetcher()
        assert fetcher._validate_url("http://example.com") is True

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        fetcher = ContentFetcher()
        assert fetcher._validate_url("https://example.com/path") is True

    def test_invalid_url_no_scheme(self):
        """Test URL without scheme."""
        fetcher = ContentFetcher()
        assert fetcher._validate_url("example.com") is False

    def test_invalid_url_no_host(self):
        """Test URL without host."""
        fetcher = ContentFetcher()
        assert fetcher._validate_url("https://") is False

    def test_invalid_url_ftp(self):
        """Test non-HTTP URL."""
        fetcher = ContentFetcher()
        assert fetcher._validate_url("ftp://files.example.com") is False


class TestFetch:
    """Tests for fetch functionality."""

    def test_basic_fetch(self, mock_response, sample_markdown_content, clean_cache):
        """Test basic fetch returns properly formatted results."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com/article")

            assert result["success"] is True
            assert result["url"] == "https://example.com/article"
            assert result["title"] == "Sample Article Title"
            assert "introduction paragraph" in result["content"]
            assert result["author"] == "John Doe"
            assert result["word_count"] > 0
            assert result["fetched_at"]

    def test_fetch_extracts_excerpt(self, mock_response, clean_cache):
        """Test that excerpt is properly extracted."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com/article")

            assert len(result["excerpt"]) <= 503  # 500 + "..."
            assert result["excerpt"]

    def test_empty_url_returns_error(self):
        """Test empty URL returns error result."""
        fetcher = ContentFetcher()

        result = fetcher.fetch("")
        assert result["success"] is False
        assert "Empty URL" in result["error"]

        result = fetcher.fetch("   ")
        assert result["success"] is False

    def test_invalid_url_returns_error(self):
        """Test invalid URL returns error result."""
        fetcher = ContentFetcher()

        result = fetcher.fetch("not-a-url")
        assert result["success"] is False
        assert "Invalid URL" in result["error"]

    def test_max_length_truncation(self, clean_cache):
        """Test content truncation at max_length."""
        long_content = "# Title\n\n" + "This is a paragraph. " * 1000
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = long_content

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com", max_length=500)

            assert result["success"] is True
            assert len(result["content"]) <= 550  # Some buffer for truncation message
            assert result["truncated"] is True
            assert "[Content truncated...]" in result["content"]


class TestCaching:
    """Tests for cache functionality."""

    def test_cache_hit(self, mock_response, clean_cache):
        """Test second identical fetch uses cache."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response) as mock_get:
            fetcher = ContentFetcher()

            # First call
            result1 = fetcher.fetch("https://example.com/article")
            assert mock_get.call_count == 1

            # Second call should use cache
            result2 = fetcher.fetch("https://example.com/article")
            assert mock_get.call_count == 1  # No additional call

            assert result1["url"] == result2["url"]
            assert result1["content"] == result2["content"]

    def test_cache_bypass(self, mock_response, clean_cache):
        """Test use_cache=False bypasses cache."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response) as mock_get:
            fetcher = ContentFetcher()

            fetcher.fetch("https://example.com", use_cache=True)
            fetcher.fetch("https://example.com", use_cache=False)

            assert mock_get.call_count == 2

    def test_cache_different_urls(self, mock_response, clean_cache):
        """Test different URLs don't share cache."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response) as mock_get:
            fetcher = ContentFetcher()

            fetcher.fetch("https://example.com/page1")
            fetcher.fetch("https://example.com/page2")

            assert mock_get.call_count == 2

    def test_clear_cache(self, mock_response, clean_cache):
        """Test cache clearing."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            fetcher = ContentFetcher()

            # Create cache entries
            fetcher.fetch("https://example.com/1")
            fetcher.fetch("https://example.com/2")
            fetcher.fetch("https://example.com/3")

            # Clear all
            cleared = fetcher.clear_cache()
            assert cleared == 3

            # Verify empty
            stats = fetcher.get_cache_stats()
            assert stats["total_entries"] == 0

    def test_cache_stats(self, mock_response, clean_cache):
        """Test cache statistics."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            fetcher = ContentFetcher()

            # Empty cache
            stats = fetcher.get_cache_stats()
            assert stats["total_entries"] == 0

            # Add entries
            fetcher.fetch("https://example.com/1")
            fetcher.fetch("https://example.com/2")

            stats = fetcher.get_cache_stats()
            assert stats["total_entries"] == 2
            assert stats["total_size_kb"] > 0


class TestErrorHandling:
    """Tests for error handling and retries."""

    def test_404_returns_error(self, clean_cache):
        """Test 404 response returns error result."""
        mock_resp = Mock()
        mock_resp.status_code = 404

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com/notfound")

            assert result["success"] is False
            assert "404" in result["error"]

    def test_500_retries(self, mock_response, clean_cache):
        """Test 500 errors trigger retries."""
        mock_500 = Mock()
        mock_500.status_code = 500

        with patch(REQUESTS_GET_PATCH) as mock_get:
            mock_get.side_effect = [mock_500, mock_500, mock_response]

            with patch(TIME_SLEEP_PATCH):
                fetcher = ContentFetcher()
                result = fetcher.fetch("https://example.com")

            assert result["success"] is True
            assert mock_get.call_count == 3

    def test_rate_limit_retries(self, mock_response, clean_cache):
        """Test rate limit (429) triggers retries."""
        mock_429 = Mock()
        mock_429.status_code = 429

        with patch(REQUESTS_GET_PATCH) as mock_get:
            mock_get.side_effect = [mock_429, mock_429, mock_response]

            with patch(TIME_SLEEP_PATCH):
                fetcher = ContentFetcher()
                result = fetcher.fetch("https://example.com")

            assert result["success"] is True

    def test_max_retries_exceeded(self, clean_cache):
        """Test failure after max retries."""
        mock_500 = Mock()
        mock_500.status_code = 500

        with patch(REQUESTS_GET_PATCH, return_value=mock_500):
            with patch(TIME_SLEEP_PATCH):
                fetcher = ContentFetcher()
                result = fetcher.fetch("https://example.com")

            assert result["success"] is False

    def test_timeout_handling(self, clean_cache):
        """Test timeout exception handling."""
        with patch(REQUESTS_GET_PATCH) as mock_get:
            mock_get.side_effect = requests_lib.Timeout("Connection timed out")

            with patch(TIME_SLEEP_PATCH):
                fetcher = ContentFetcher()
                result = fetcher.fetch("https://example.com")

            assert result["success"] is False
            assert "Timeout" in result["error"] or "timed out" in result["error"].lower()

    def test_connection_error_retries(self, mock_response, clean_cache):
        """Test connection errors trigger retries."""
        with patch(REQUESTS_GET_PATCH) as mock_get:
            mock_get.side_effect = [
                requests_lib.ConnectionError("Network unreachable"),
                mock_response,
            ]

            with patch(TIME_SLEEP_PATCH):
                fetcher = ContentFetcher()
                result = fetcher.fetch("https://example.com")

            assert result["success"] is True

    def test_empty_content_returns_error(self, clean_cache):
        """Test empty content is handled as error."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = "   "

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com")

            assert result["success"] is False
            assert "No meaningful content" in result["error"]


class TestContentParsing:
    """Tests for content parsing."""

    def test_title_from_h1(self, clean_cache):
        """Test title extraction from H1 header."""
        content = "# My Great Article\n\nContent here and more content to make it long enough."
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = content

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com")

            assert result["title"] == "My Great Article"

    def test_title_fallback_to_url(self, clean_cache):
        """Test title fallback to URL when no H1."""
        content = "Just some content without a title header. " * 10
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = content

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com/my-article-slug")

            assert result["title"] == "My Article Slug"

    def test_author_extraction(self, clean_cache):
        """Test author extraction from content."""
        content = "# Title\n\nAuthor: Jane Smith\n\nArticle content goes here. " * 5
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = content

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com")

            assert result["author"] == "Jane Smith"

    def test_word_count(self, clean_cache):
        """Test word count calculation."""
        content = "# Title\n\n" + "word " * 100
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.text = content

        with patch(REQUESTS_GET_PATCH, return_value=mock_resp):
            fetcher = ContentFetcher()
            result = fetcher.fetch("https://example.com")

            assert result["word_count"] >= 100


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_spacing(self, mock_response, clean_cache):
        """Test requests are spaced appropriately."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            with patch(TIME_SLEEP_PATCH) as mock_sleep:
                fetcher = ContentFetcher()

                fetcher.fetch("https://example.com/1", use_cache=False)
                fetcher.fetch("https://example.com/2", use_cache=False)

                # Second request should trigger rate limit wait
                assert mock_sleep.called


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_fetch_url(self, mock_response, clean_cache):
        """Test fetch_url convenience function."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            result = fetch_url("https://example.com/article")

            assert result["success"] is True
            assert result["url"] == "https://example.com/article"

    def test_fetch_urls(self, mock_response, clean_cache):
        """Test fetch_urls convenience function."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            urls = [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
            ]
            results = fetch_urls(urls)

            assert len(results) == 3
            assert all(r["success"] for r in results)

    def test_fetch_multiple(self, mock_response, clean_cache):
        """Test fetching multiple URLs."""
        with patch(REQUESTS_GET_PATCH, return_value=mock_response):
            fetcher = ContentFetcher()
            urls = ["https://example.com/a", "https://example.com/b"]
            results = fetcher.fetch_multiple(urls)

            assert len(results) == 2
            assert all(r["success"] for r in results)


# Integration Tests
class TestIntegration:
    """Integration tests that hit real endpoints."""

    def test_fetch_wikipedia(self, clean_cache):
        """Test fetching from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        result = fetch_url(url)

        assert result["success"] is True
        assert len(result["content"]) > 100
        assert result["title"]
        assert result["word_count"] > 50

    def test_fetch_example_com(self, clean_cache):
        """Test fetching example.com."""
        result = fetch_url("https://example.com")

        assert result["success"] is True
        assert result["content"]

    def test_fetch_invalid_domain(self, clean_cache):
        """Test fetching from non-existent domain."""
        result = fetch_url("https://this-domain-definitely-does-not-exist-12345.com")

        assert result["success"] is False
        assert result["error"]

    def test_cache_speeds_up_second_fetch(self, clean_cache):
        """Test that caching improves performance."""
        url = "https://example.com"

        # First fetch
        start1 = time.time()
        result1 = fetch_url(url)
        time1 = time.time() - start1

        # Second fetch (cached)
        start2 = time.time()
        result2 = fetch_url(url)
        time2 = time.time() - start2

        assert result1["content"] == result2["content"]
        assert time2 < time1  # Cached should be faster

    def test_max_length_real_page(self, clean_cache):
        """Test max_length truncation on real page."""
        result = fetch_url(
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            max_length=1000,
        )

        assert result["success"] is True
        assert len(result["content"]) <= 1100  # Buffer for truncation message
