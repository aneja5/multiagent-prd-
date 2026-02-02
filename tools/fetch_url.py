"""
Fetch and clean web page content using Jina Reader.

This module provides robust URL content fetching for the ResearcherAgent,
including caching, retry logic, rate limiting, and comprehensive error handling.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib
import json
import time
import os
from pathlib import Path
from functools import wraps

import requests

from app.logger import get_logger

logger = get_logger(__name__)

# Cache configuration
CACHE_DIR = Path("data/cache/content")
CACHE_TTL_HOURS = 48  # Content cache longer than search results

# Jina Reader configuration
JINA_READER_BASE = "https://r.jina.ai"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 30.0

# Rate limit configuration
MIN_REQUEST_INTERVAL_SECONDS = 0.5

# Request configuration
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_LENGTH = 10000


class ContentFetchError(Exception):
    """Base exception for content fetch errors."""
    pass


class RateLimitError(ContentFetchError):
    """Raised when rate limit is exceeded."""
    pass


class NetworkError(ContentFetchError):
    """Raised for network-related errors."""
    pass


class ContentValidationError(ContentFetchError):
    """Raised when content validation fails."""
    pass


def retry_with_backoff(max_retries: int = MAX_RETRIES):
    """
    Decorator for retry logic with exponential backoff.

    Handles transient failures and rate limits gracefully.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            backoff = INITIAL_BACKOFF_SECONDS

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = min(backoff * 2, MAX_BACKOFF_SECONDS)
                        logger.warning(
                            f"Rate limited, waiting {sleep_time:.1f}s before retry "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        backoff *= BACKOFF_MULTIPLIER
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries")
                        raise
                except NetworkError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Network error: {str(e)}, retrying in {backoff:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(backoff)
                        backoff *= BACKOFF_MULTIPLIER
                    else:
                        logger.error(f"Network error after {max_retries} retries: {str(e)}")
                        raise
                except ContentFetchError:
                    # Non-retryable errors
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in fetch: {str(e)}")
                    raise ContentFetchError(f"Fetch failed: {str(e)}") from e

            raise last_exception if last_exception else ContentFetchError("Fetch failed")

        return wrapper
    return decorator


class ContentFetcher:
    """
    Fetch clean content from URLs using Jina Reader.

    Provides reliable content fetching with caching, retry logic,
    and rate limiting for production use.
    """

    def __init__(
        self,
        base_url: str = JINA_READER_BASE,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the content fetcher.

        Args:
            base_url: Jina Reader base URL
            timeout: Request timeout in seconds
            api_key: Optional Jina API key for higher rate limits
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self._last_request_time: Optional[float] = None

        # Ensure cache directory exists
        self._ensure_cache_dir()

        logger.debug(f"ContentFetcher initialized with base_url={base_url}")

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory ready: {CACHE_DIR}")
        except OSError as e:
            logger.warning(f"Could not create cache directory: {e}")

    def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between API requests."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                wait_time = MIN_REQUEST_INTERVAL_SECONDS - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        self._last_request_time = time.time()

    def _validate_url(self, url: str) -> bool:
        """
        Validate URL format.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            return all([parsed.scheme in ("http", "https"), parsed.netloc])
        except Exception:
            return False

    def fetch(
        self,
        url: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch and clean content from URL.

        Args:
            url: URL to fetch
            max_length: Maximum content length (chars)
            use_cache: Whether to use cached results (default: True)

        Returns:
            Dict with:
            - url: str (original URL)
            - title: str
            - content: str (clean markdown)
            - author: Optional[str]
            - published_date: Optional[str]
            - excerpt: str (first 500 chars)
            - word_count: int
            - fetched_at: str (ISO timestamp)
            - success: bool
            - error: Optional[str] (if success is False)
        """
        # Validate URL
        if not url or not url.strip():
            logger.warning("Empty URL provided")
            return self._error_result(url, "Empty URL")

        url = url.strip()

        if not self._validate_url(url):
            logger.warning(f"Invalid URL format: {url}")
            return self._error_result(url, "Invalid URL format")

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(url)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for URL: {url[:50]}...")
                return cached

        logger.info(f"Fetching: {url[:60]}{'...' if len(url) > 60 else ''}")

        try:
            result = self._execute_fetch(url, max_length)

            # Cache successful results
            if use_cache and result.get("success"):
                self._save_to_cache(cache_key, result)

            return result

        except ContentFetchError as e:
            logger.error(f"Fetch failed for {url}: {e}")
            return self._error_result(url, str(e))

    @retry_with_backoff(max_retries=MAX_RETRIES)
    def _execute_fetch(self, url: str, max_length: int) -> Dict[str, Any]:
        """
        Execute the actual fetch with retry logic.

        This method is decorated with retry_with_backoff for resilience.
        """
        # Enforce rate limiting
        self._rate_limit_wait()

        try:
            # Build Jina Reader URL
            jina_url = f"{self.base_url}/{url}"

            # Build headers
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
                "Accept": "text/plain, text/markdown",
            }

            # Add API key if available
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            logger.debug(f"Requesting: {jina_url}")

            response = requests.get(
                jina_url,
                timeout=self.timeout,
                headers=headers,
            )

            # Handle different status codes
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")

            if response.status_code == 404:
                logger.warning(f"URL not found: {url}")
                return self._error_result(url, "Page not found (404)")

            if response.status_code >= 500:
                raise NetworkError(f"Server error: HTTP {response.status_code}")

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return self._error_result(url, f"HTTP {response.status_code}")

            # Parse content
            content = response.text

            # Validate content
            if not content or len(content.strip()) < 50:
                logger.warning(f"Empty or minimal content from {url}")
                return self._error_result(url, "No meaningful content found")

            # Extract metadata from content
            result = self._parse_content(url, content, max_length)

            logger.info(f"Fetched {result['word_count']} words from {url[:40]}...")

            return result

        except requests.Timeout:
            raise NetworkError(f"Timeout after {self.timeout}s")

        except requests.ConnectionError as e:
            raise NetworkError(f"Connection failed: {str(e)}")

        except (RateLimitError, NetworkError, ContentFetchError):
            raise

        except Exception as e:
            raise ContentFetchError(f"Unexpected error: {str(e)}") from e

    def _parse_content(
        self,
        url: str,
        content: str,
        max_length: int,
    ) -> Dict[str, Any]:
        """
        Parse and structure the fetched content.

        Args:
            url: Original URL
            content: Raw markdown content from Jina
            max_length: Maximum content length

        Returns:
            Structured result dictionary
        """
        lines = content.split("\n")

        # Extract title (usually first # header)
        title = ""
        content_start_idx = 0

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# ") and not title:
                title = stripped[2:].strip()
                content_start_idx = idx + 1
                break
            elif stripped.startswith("Title:"):
                title = stripped[6:].strip()
                content_start_idx = idx + 1
                break

        # Fallback title from URL
        if not title:
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")
            if path_parts and path_parts[-1]:
                # Convert slug to title
                title = path_parts[-1].replace("-", " ").replace("_", " ").title()
            else:
                title = parsed.netloc

        # Extract author if present
        author = None
        for line in lines[:20]:  # Check first 20 lines
            if line.lower().startswith("author:") or line.lower().startswith("by "):
                author = line.split(":", 1)[-1].strip() if ":" in line else line[3:].strip()
                break

        # Extract published date if present
        published_date = None
        date_keywords = ["published:", "date:", "posted:"]
        for line in lines[:20]:
            line_lower = line.lower()
            for keyword in date_keywords:
                if keyword in line_lower:
                    published_date = line.split(":", 1)[-1].strip()
                    break
            if published_date:
                break

        # Clean content (remove excessive whitespace)
        clean_lines = []
        prev_empty = False
        for line in lines[content_start_idx:]:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            clean_lines.append(line)
            prev_empty = is_empty

        clean_content = "\n".join(clean_lines).strip()

        # Truncate if too long
        truncated = False
        if len(clean_content) > max_length:
            clean_content = clean_content[:max_length]
            # Try to break at a sentence or paragraph
            last_para = clean_content.rfind("\n\n")
            last_sentence = max(
                clean_content.rfind(". "),
                clean_content.rfind(".\n"),
            )
            break_point = max(last_para, last_sentence)
            if break_point > max_length * 0.7:  # Only if we keep 70%+
                clean_content = clean_content[:break_point + 1]
            clean_content += "\n\n[Content truncated...]"
            truncated = True

        # Calculate word count
        word_count = len(clean_content.split())

        # Generate excerpt
        excerpt = clean_content[:500]
        if len(clean_content) > 500:
            # Break at word boundary
            last_space = excerpt.rfind(" ")
            if last_space > 400:
                excerpt = excerpt[:last_space] + "..."

        return {
            "url": url,
            "title": title,
            "content": clean_content,
            "excerpt": excerpt,
            "author": author,
            "published_date": published_date,
            "word_count": word_count,
            "truncated": truncated,
            "fetched_at": datetime.now().isoformat(),
            "success": True,
        }

    def _error_result(self, url: str, error: str) -> Dict[str, Any]:
        """
        Return standardized error result.

        Args:
            url: Original URL
            error: Error message

        Returns:
            Error result dictionary
        """
        return {
            "url": url,
            "title": "",
            "content": "",
            "excerpt": "",
            "author": None,
            "published_date": None,
            "word_count": 0,
            "truncated": False,
            "fetched_at": datetime.now().isoformat(),
            "success": False,
            "error": error,
        }

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve from cache if not expired.

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached result or None if not found/expired
        """
        cache_file = CACHE_DIR / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            # Check if expired
            modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            age = datetime.now() - modified_time

            if age > timedelta(hours=CACHE_TTL_HOURS):
                logger.debug(f"Cache expired (age: {age}): {cache_key}")
                self._delete_cache_file(cache_file)
                return None

            # Load from cache
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.debug(f"Cache hit, age: {age}")
            return data

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error reading cache file: {e}")
            self._delete_cache_file(cache_file)
            return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Save result to cache.

        Args:
            cache_key: The cache key
            result: Result to cache
        """
        cache_file = CACHE_DIR / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cached content: {cache_key}")
        except OSError as e:
            logger.warning(f"Failed to write cache file: {e}")

    def _delete_cache_file(self, cache_file: Path) -> None:
        """Safely delete a cache file."""
        try:
            cache_file.unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"Could not delete cache file: {e}")

    def fetch_multiple(
        self,
        urls: List[str],
        max_length: int = DEFAULT_MAX_LENGTH,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch content from multiple URLs.

        Args:
            urls: List of URLs to fetch
            max_length: Maximum content length per URL
            use_cache: Whether to use cached results

        Returns:
            List of fetch results (maintains order)
        """
        results = []
        for url in urls:
            result = self.fetch(url, max_length=max_length, use_cache=use_cache)
            results.append(result)
        return results

    def clear_cache(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clear cached content.

        Args:
            max_age_hours: If provided, only clear entries older than this.
                          If None, clear all cache entries.

        Returns:
            Number of cache entries cleared
        """
        if not CACHE_DIR.exists():
            return 0

        cleared = 0
        cutoff_time = None

        if max_age_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                if cutoff_time:
                    modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if modified_time >= cutoff_time:
                        continue

                cache_file.unlink()
                cleared += 1
            except OSError as e:
                logger.debug(f"Could not delete {cache_file}: {e}")

        logger.info(f"Cleared {cleared} content cache entries")
        return cleared

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the content cache.

        Returns:
            Dictionary with cache statistics
        """
        if not CACHE_DIR.exists():
            return {
                "total_entries": 0,
                "total_size_kb": 0,
                "oldest_entry": None,
                "newest_entry": None,
            }

        entries = list(CACHE_DIR.glob("*.json"))

        if not entries:
            return {
                "total_entries": 0,
                "total_size_kb": 0,
                "oldest_entry": None,
                "newest_entry": None,
            }

        total_size = sum(f.stat().st_size for f in entries)
        timestamps = [datetime.fromtimestamp(f.stat().st_mtime) for f in entries]

        return {
            "total_entries": len(entries),
            "total_size_kb": round(total_size / 1024, 2),
            "oldest_entry": min(timestamps).isoformat(),
            "newest_entry": max(timestamps).isoformat(),
        }


def fetch_url(
    url: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to fetch URL content.

    Creates a ContentFetcher instance and fetches the URL.
    For multiple fetches, prefer creating a ContentFetcher instance
    directly to benefit from rate limiting state.

    Args:
        url: URL to fetch
        max_length: Maximum content length (chars)
        use_cache: Whether to use cached results

    Returns:
        Dict with url, title, content, excerpt, success, and optional error
    """
    fetcher = ContentFetcher()
    return fetcher.fetch(url, max_length=max_length, use_cache=use_cache)


def fetch_urls(
    urls: List[str],
    max_length: int = DEFAULT_MAX_LENGTH,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch multiple URLs.

    Args:
        urls: List of URLs to fetch
        max_length: Maximum content length per URL
        use_cache: Whether to use cached results

    Returns:
        List of fetch results
    """
    fetcher = ContentFetcher()
    return fetcher.fetch_multiple(urls, max_length=max_length, use_cache=use_cache)
