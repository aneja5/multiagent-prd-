"""
Web search using Tavily API with caching and error handling.

This module provides robust web search capabilities for the ResearcherAgent,
including caching, retry logic, rate limiting, and comprehensive error handling.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import os
import time
import hashlib
from pathlib import Path
from functools import wraps

from tavily import TavilyClient

from app.logger import get_logger

logger = get_logger(__name__)

# Cache configuration
CACHE_DIR = Path("data/cache/search")
CACHE_TTL_HOURS = 24  # Cache results for 24 hours

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 30.0

# Rate limit configuration
MIN_REQUEST_INTERVAL_SECONDS = 0.5  # Minimum time between requests


class WebSearchError(Exception):
    """Base exception for web search errors."""
    pass


class RateLimitError(WebSearchError):
    """Raised when rate limit is exceeded."""
    pass


class SearchAPIError(WebSearchError):
    """Raised when the search API returns an error."""
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
                except (ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Connection error: {str(e)}, retrying in {backoff:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(backoff)
                        backoff *= BACKOFF_MULTIPLIER
                    else:
                        logger.error(f"Connection failed after {max_retries} retries: {str(e)}")
                        raise SearchAPIError(f"Connection failed: {str(e)}") from e
                except Exception as e:
                    # For unexpected errors, log and re-raise without retry
                    logger.error(f"Unexpected error in search: {str(e)}")
                    raise SearchAPIError(f"Search failed: {str(e)}") from e

            # Should not reach here, but handle it
            raise last_exception if last_exception else SearchAPIError("Search failed")

        return wrapper
    return decorator


class WebSearchTool:
    """
    Web search with caching, deduplication, and error handling.

    Provides reliable web search capabilities using the Tavily API,
    with built-in caching to reduce API calls and improve response times.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.

        Args:
            api_key: Tavily API key. If not provided, reads from TAVILY_API_KEY env var.

        Raises:
            ValueError: If no API key is found.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.error("TAVILY_API_KEY not found in environment")
            raise ValueError("TAVILY_API_KEY not found in environment")

        self.client = TavilyClient(api_key=self.api_key)
        self._last_request_time: Optional[float] = None

        # Ensure cache directory exists
        self._ensure_cache_dir()

        logger.debug("WebSearchTool initialized successfully")

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

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search the web and return clean results.

        Args:
            query: Search query string
            max_results: Maximum number of results (default: 10)
            search_depth: "basic" (faster) or "advanced" (more comprehensive)
            include_domains: List of domains to prioritize
            exclude_domains: List of domains to exclude
            use_cache: Whether to use cached results (default: True)

        Returns:
            List of search results with:
            - url: str
            - title: str
            - snippet: str (clean, relevant excerpt)
            - published_date: Optional[str]
            - score: float (relevance score 0-1)

        Raises:
            WebSearchError: If search fails after all retries
        """
        # Validate input
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        query = query.strip()

        # Validate search_depth
        if search_depth not in ("basic", "advanced"):
            logger.warning(f"Invalid search_depth '{search_depth}', defaulting to 'advanced'")
            search_depth = "advanced"

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(query, max_results, search_depth, include_domains, exclude_domains)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result

        logger.info(f"Searching: {query[:50]}{'...' if len(query) > 50 else ''}")

        # Execute search with retry logic
        results = self._execute_search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

        # Cache results
        if use_cache and results:
            self._save_to_cache(cache_key, results)

        return results

    @retry_with_backoff(max_retries=MAX_RETRIES)
    def _execute_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Execute the actual search API call with retry logic.

        This method is decorated with retry_with_backoff for resilience.
        """
        # Enforce rate limiting
        self._rate_limit_wait()

        try:
            logger.debug(f"Calling Tavily API: depth={search_depth}, max_results={max_results}")

            # Build API call parameters
            api_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_raw_content": False,
            }

            if include_domains:
                api_params["include_domains"] = include_domains
                logger.debug(f"Including domains: {include_domains}")

            if exclude_domains:
                api_params["exclude_domains"] = exclude_domains
                logger.debug(f"Excluding domains: {exclude_domains}")

            # Call Tavily API
            response = self.client.search(**api_params)

            # Parse and validate response
            results = self._parse_response(response)

            logger.info(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit hit: {e}")
                raise RateLimitError(f"Rate limit exceeded: {e}") from e

            # Check for authentication errors
            if "401" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
                logger.error(f"Authentication failed: {e}")
                raise SearchAPIError(f"Authentication failed: {e}") from e

            # Re-raise for retry decorator to handle
            raise

    def _parse_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse Tavily API response into clean result format.

        Args:
            response: Raw API response

        Returns:
            List of parsed search results
        """
        results = []

        raw_results = response.get("results", [])
        logger.debug(f"Parsing {len(raw_results)} raw results")

        for item in raw_results:
            try:
                result = {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "published_date": item.get("published_date"),
                    "score": float(item.get("score", 0.5)),
                }

                # Validate required fields
                if result["url"] and result["title"]:
                    results.append(result)
                else:
                    logger.debug(f"Skipping result with missing url/title: {item}")

            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing result item: {e}")
                continue

        return results

    def _get_cache_key(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
    ) -> str:
        """
        Generate deterministic cache key from search parameters.

        Args:
            query: Search query
            max_results: Maximum results requested
            search_depth: Search depth setting
            include_domains: Domains to include
            exclude_domains: Domains to exclude

        Returns:
            MD5 hash string as cache key
        """
        # Create deterministic string representation
        cache_data = {
            "query": query.lower(),
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": sorted(include_domains) if include_domains else None,
            "exclude_domains": sorted(exclude_domains) if exclude_domains else None,
        }
        content = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve results from cache if not expired.

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached results or None if not found/expired
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

    def _save_to_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """
        Save results to cache.

        Args:
            cache_key: The cache key
            results: Results to cache
        """
        cache_file = CACHE_DIR / f"{cache_key}.json"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cached {len(results)} results: {cache_key}")
        except OSError as e:
            logger.warning(f"Failed to write cache file: {e}")

    def _delete_cache_file(self, cache_file: Path) -> None:
        """Safely delete a cache file."""
        try:
            cache_file.unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"Could not delete cache file: {e}")

    def clear_cache(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clear cached search results.

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

        logger.info(f"Cleared {cleared} cache entries")
        return cleared

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics:
            - total_entries: Number of cached searches
            - total_size_kb: Total size in kilobytes
            - oldest_entry: Timestamp of oldest entry
            - newest_entry: Timestamp of newest entry
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


def search_web(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function for web search.

    Creates a WebSearchTool instance and performs a search.
    For multiple searches, prefer creating a WebSearchTool instance directly
    to benefit from connection reuse and rate limiting.

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 10)
        search_depth: "basic" (faster) or "advanced" (more comprehensive)
        include_domains: List of domains to prioritize
        exclude_domains: List of domains to exclude

    Returns:
        List of search results with url, title, snippet, published_date, score
    """
    try:
        tool = WebSearchTool()
        return tool.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    except ValueError as e:
        logger.error(f"WebSearchTool initialization failed: {e}")
        return []
    except WebSearchError as e:
        logger.error(f"Search failed: {e}")
        return []
