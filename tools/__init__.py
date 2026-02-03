"""
Tools package for the multi-agent PRD generator.

Contains utilities for web search, content extraction, credibility scoring,
deduplication, and other research tools.
"""

from tools.web_search import WebSearchTool, search_web
from tools.fetch_url import ContentFetcher, fetch_url, fetch_urls
from tools.credibility import CredibilityScorer, score_credibility, score_sources
from tools.dedupe import Deduplicator, deduplicate_evidence, merge_evidence

__all__ = [
    # Web search
    "WebSearchTool",
    "search_web",
    # Content fetching
    "ContentFetcher",
    "fetch_url",
    "fetch_urls",
    # Credibility scoring
    "CredibilityScorer",
    "score_credibility",
    "score_sources",
    # Deduplication
    "Deduplicator",
    "deduplicate_evidence",
    "merge_evidence",
]
