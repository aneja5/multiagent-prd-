"""
Deduplicate evidence based on URL and content similarity.

This module provides deduplication capabilities to remove redundant
evidence entries from research results, improving PRD quality.

Uses a hybrid approach:
- MD5 hash for exact content matches (O(1) lookup)
- SimHash for near-duplicate detection (catches paraphrased/syndicated content)
- Fuzzy string matching for titles and URLs
"""

from typing import List, Dict, Set, Tuple, Optional
from urllib.parse import urlparse, parse_qs, urlunparse, unquote
import hashlib
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from simhash import Simhash

from app.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SimHash Configuration
# =============================================================================

# Number of bits difference allowed for SimHash to be considered similar
# 3 bits = ~95% similar, 6 bits = ~91% similar, 8 bits = ~87% similar
# 6 is a good balance - catches syndicated content while avoiding false positives
SIMHASH_DISTANCE_THRESHOLD = 6

# Minimum content length (in characters) to compute SimHash
# SimHash works best on longer text
SIMHASH_MIN_CONTENT_LENGTH = 100


# Common tracking parameters to remove during canonicalization
TRACKING_PARAMS: Set[str] = {
    # UTM parameters
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "utm_id",
    "utm_cid",
    # Common referral/tracking
    "ref",
    "source",
    "src",
    "referrer",
    "referer",
    # Social media
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "twclid",
    "li_fat_id",
    # Analytics
    "_ga",
    "_gl",
    "mc_cid",
    "mc_eid",
    # Session/user tracking
    "sessionid",
    "session_id",
    "sid",
    "uid",
    "userid",
    # Affiliate
    "affiliate",
    "aff_id",
    "partner",
    # Misc tracking
    "tracking",
    "track",
    "click",
    "clickid",
    "campaign",
    "ad",
    "adid",
}

# URL path suffixes that are often equivalent
EQUIVALENT_SUFFIXES: List[Tuple[str, str]] = [
    ("/index.html", "/"),
    ("/index.htm", "/"),
    ("/index.php", "/"),
    ("/default.html", "/"),
    ("/default.aspx", "/"),
]


class Deduplicator:
    """
    Deduplicate evidence entries based on URL and content similarity.

    Uses a hybrid approach with multiple strategies:
    1. URL canonicalization (normalize URLs for comparison)
    2. Exact URL matching (fast set lookup)
    3. Fuzzy URL matching (for redirects, minor variations)
    4. MD5 hash for exact content matches (fast set lookup)
    5. SimHash for near-duplicate content (catches paraphrased/syndicated)
    6. Fuzzy title matching (for same article, different URLs)
    """

    def __init__(
        self,
        url_similarity_threshold: int = 90,
        content_similarity_threshold: int = 85,
        title_similarity_threshold: int = 90,
        simhash_threshold: int = SIMHASH_DISTANCE_THRESHOLD,
    ):
        """
        Initialize the deduplicator.

        Args:
            url_similarity_threshold: Minimum % similarity for URL fuzzy match (0-100)
            content_similarity_threshold: Minimum % similarity for content match (0-100)
            title_similarity_threshold: Minimum % similarity for title-only match (0-100)
            simhash_threshold: Max Hamming distance for SimHash match (default: 5 bits)
        """
        self.url_similarity_threshold = url_similarity_threshold
        self.content_similarity_threshold = content_similarity_threshold
        self.title_similarity_threshold = title_similarity_threshold
        self.simhash_threshold = simhash_threshold

    def deduplicate(
        self,
        evidence_list: List[Dict],
        preserve_order: bool = True,
    ) -> List[Dict]:
        """
        Remove duplicate evidence entries.

        Strategy (in order, fast checks first):
        1. Canonicalize URLs (remove tracking params, fragments, normalize domain)
        2. Check exact URL matches (O(1) set lookup)
        3. Check fuzzy URL similarity (for redirects, www vs non-www)
        4. Check MD5 content hash (O(1) exact content match)
        5. Check SimHash similarity (near-duplicate detection for paraphrased content)
        6. Check fuzzy title match (for same article, different URLs)

        Args:
            evidence_list: List of evidence dictionaries with url, title, snippet
            preserve_order: If True, keeps first occurrence; if False, may reorder

        Returns:
            Deduplicated list (keeps first occurrence of each unique item)
        """
        if not evidence_list:
            return []

        original_count = len(evidence_list)
        seen_canonical_urls: Set[str] = set()
        seen_url_variations: List[str] = []  # For fuzzy matching
        seen_content_hashes: Set[str] = set()  # MD5 hashes for exact match
        seen_simhashes: List[Tuple[Simhash, int]] = []  # (simhash, index) for near-duplicate
        seen_titles: List[Tuple[str, int]] = []  # (title, index) for fuzzy title matching
        unique_evidence: List[Dict] = []

        # Track duplicate reasons for logging
        dup_reasons = {"url_exact": 0, "url_fuzzy": 0, "md5": 0, "simhash": 0, "title": 0}

        for idx, item in enumerate(evidence_list):
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", item.get("content", ""))

            # Skip items without URLs
            if not url:
                logger.debug(f"Skipping item {idx} with no URL")
                continue

            # Canonicalize URL
            canonical_url = self._canonicalize_url(url)

            # Check 1: Exact canonical URL match (fast O(1))
            if canonical_url in seen_canonical_urls:
                logger.debug(f"Duplicate (exact URL): {url[:60]}...")
                dup_reasons["url_exact"] += 1
                continue

            # Check 2: Fuzzy URL match
            if self._is_url_similar_to_seen(canonical_url, seen_url_variations):
                logger.debug(f"Duplicate (similar URL): {url[:60]}...")
                dup_reasons["url_fuzzy"] += 1
                continue

            # Check 3: MD5 content hash match (fast O(1) exact content duplicate)
            content_hash = self._get_content_hash(title, snippet)
            if content_hash is not None and content_hash in seen_content_hashes:
                logger.debug(f"Duplicate (exact content MD5): {(title or '')[:40]}...")
                dup_reasons["md5"] += 1
                continue

            # Check 4: SimHash similarity (near-duplicate detection)
            content_simhash = self._compute_simhash(title, snippet)
            if content_simhash is not None:
                similar_idx = self._find_similar_simhash(content_simhash, seen_simhashes)
                if similar_idx is not None:
                    logger.debug(
                        f"Duplicate (SimHash near-match): {(title or '')[:40]}... "
                        f"similar to item {similar_idx}"
                    )
                    dup_reasons["simhash"] += 1
                    continue

            # Check 5: Fuzzy title match (for same article, different URLs)
            if title and self._is_title_similar_to_seen(title, seen_titles):
                logger.debug(f"Duplicate (similar title): {title[:40]}...")
                dup_reasons["title"] += 1
                continue

            # It's unique - add to all tracking sets
            seen_canonical_urls.add(canonical_url)
            seen_url_variations.append(canonical_url)
            if content_hash is not None:
                seen_content_hashes.add(content_hash)
            if content_simhash is not None:
                seen_simhashes.append((content_simhash, len(unique_evidence)))
            if title:
                seen_titles.append((title.lower(), len(unique_evidence)))

            unique_evidence.append(item)

        duplicate_count = sum(dup_reasons.values())
        if duplicate_count > 0:
            logger.info(
                f"Deduplicated {original_count} -> {len(unique_evidence)} "
                f"(removed {duplicate_count}: url={dup_reasons['url_exact']+dup_reasons['url_fuzzy']}, "
                f"md5={dup_reasons['md5']}, simhash={dup_reasons['simhash']}, title={dup_reasons['title']})"
            )

        return unique_evidence

    def _compute_simhash(self, title: str, snippet: str) -> Optional[Simhash]:
        """
        Compute SimHash for content.

        SimHash is a locality-sensitive hash that produces similar hashes
        for similar content. Two documents with Hamming distance <= threshold
        are considered near-duplicates.

        Args:
            title: Article title
            snippet: Article snippet/content

        Returns:
            Simhash object, or None if content is too short
        """
        # Normalize content
        title_str = title or ""
        snippet_str = snippet or ""
        combined = f"{title_str} {snippet_str}".lower()

        # Remove extra whitespace and normalize
        combined = " ".join(combined.split())

        # SimHash needs sufficient content to be meaningful
        if len(combined) < SIMHASH_MIN_CONTENT_LENGTH:
            return None

        try:
            # Simhash uses word-level features by default
            return Simhash(combined)
        except Exception as e:
            logger.debug(f"SimHash computation failed: {e}")
            return None

    def _find_similar_simhash(
        self,
        target: Simhash,
        seen_simhashes: List[Tuple[Simhash, int]],
    ) -> Optional[int]:
        """
        Find a similar SimHash in the seen list.

        Uses Hamming distance to compare SimHash values.
        Hamming distance = number of bit positions that differ.

        Args:
            target: SimHash to check
            seen_simhashes: List of (Simhash, original_index) tuples

        Returns:
            Index of similar item, or None if no match
        """
        for seen_hash, original_idx in seen_simhashes:
            distance = target.distance(seen_hash)
            if distance <= self.simhash_threshold:
                logger.debug(f"SimHash match: distance={distance} (threshold={self.simhash_threshold})")
                return original_idx

        return None

    def _canonicalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.

        Operations:
        - Lowercase domain
        - Remove www. prefix
        - Remove tracking parameters
        - Remove fragment (#...)
        - Normalize path (trailing slashes, index files)
        - Decode URL encoding

        Args:
            url: Original URL

        Returns:
            Canonicalized URL string
        """
        if not url:
            return ""

        try:
            # Decode URL encoding
            url = unquote(url)

            parsed = urlparse(url)

            # Normalize scheme
            scheme = parsed.scheme.lower() or "https"

            # Normalize domain
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

            # Remove port if default
            if ":" in domain:
                host, port = domain.rsplit(":", 1)
                if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                    domain = host

            # Normalize path
            path = parsed.path

            # Remove index files first
            for suffix, replacement in EQUIVALENT_SUFFIXES:
                if path.endswith(suffix):
                    path = path[:-len(suffix)] + replacement
                    break

            # Remove trailing slash (except for root) - after index file removal
            if path != "/" and path.endswith("/"):
                path = path.rstrip("/")

            # Handle empty path
            if not path:
                path = "/"

            # Clean query parameters (remove tracking)
            query_params = parse_qs(parsed.query, keep_blank_values=False)
            clean_params = {
                k: v for k, v in query_params.items()
                if k.lower() not in TRACKING_PARAMS
            }

            # Sort and rebuild query string for consistency
            if clean_params:
                sorted_params = sorted(clean_params.items())
                clean_query = "&".join(
                    f"{k}={v[0]}" for k, v in sorted_params if v
                )
            else:
                clean_query = ""

            # Rebuild URL (without fragment)
            canonical = urlunparse((
                scheme,
                domain,
                path,
                "",  # params
                clean_query,
                "",  # fragment removed
            ))

            return canonical

        except Exception as e:
            logger.debug(f"Error canonicalizing URL '{url}': {e}")
            return url.lower()

    def _is_url_similar_to_seen(
        self,
        url: str,
        seen_urls: List[str],
    ) -> bool:
        """
        Check if URL is similar to any previously seen URL.

        Uses fuzzy string matching to catch:
        - Minor typos
        - Redirect variations
        - Parameter reordering

        Args:
            url: URL to check
            seen_urls: List of previously seen canonical URLs

        Returns:
            True if similar URL was found
        """
        if not seen_urls:
            return False

        for seen in seen_urls:
            # Quick length check first (optimization)
            len_diff = abs(len(url) - len(seen))
            if len_diff > 20:  # URLs too different in length
                continue

            # Check if one is a prefix of the other (with small suffix)
            # This catches /page vs /page/ variations
            if url.startswith(seen) or seen.startswith(url):
                suffix_len = abs(len(url) - len(seen))
                if suffix_len <= 2:  # Only trailing slash or similar
                    return True

            # For fuzzy matching, only consider URLs that are:
            # 1. Long enough (> 50 chars) to avoid false positives on short URLs
            # 2. Have a very high similarity ratio (>= threshold)
            if len(url) > 50 and len(seen) > 50:
                similarity = fuzz.ratio(url, seen)
                if similarity >= self.url_similarity_threshold:
                    return True

            # Also check if paths are very similar (different domains, same content)
            # Only for paths that are substantial (>25 chars) to avoid false matches
            url_path = urlparse(url).path
            seen_path = urlparse(seen).path
            if url_path and seen_path and len(url_path) > 25 and len(seen_path) > 25:
                path_similarity = fuzz.ratio(url_path, seen_path)
                if path_similarity >= 95:  # Very similar paths
                    return True

        return False

    def _get_content_hash(self, title: str, snippet: str) -> Optional[str]:
        """
        Generate hash of content for exact duplicate detection.

        Args:
            title: Article title
            snippet: Article snippet/content

        Returns:
            MD5 hash string, or None for empty content (skips content check)
        """
        # Normalize content
        title_str = title or ""
        snippet_str = snippet or ""
        combined = f"{title_str} {snippet_str}".lower()

        # Remove extra whitespace
        combined = " ".join(combined.split())

        # Remove common punctuation variations
        combined = re.sub(r'[^\w\s]', '', combined)

        # If content is empty or very short, return None to skip content dedup
        # This prevents matching all empty content together
        if len(combined.strip()) < 10:
            return None

        return hashlib.md5(combined.encode()).hexdigest()

    def _is_title_similar_to_seen(
        self,
        title: str,
        seen_titles: List[Tuple[str, int]],
    ) -> bool:
        """
        Check if title is similar to any previously seen title.

        Catches republished content across different domains.

        Args:
            title: Title to check
            seen_titles: List of (lowercase_title, index) tuples

        Returns:
            True if similar title was found
        """
        if not title or not seen_titles:
            return False

        title_lower = title.lower().strip()

        # Don't match very short titles (too many false positives)
        if len(title_lower) < 10:
            return False

        for seen_title, _ in seen_titles:
            # Skip empty/short seen titles
            if len(seen_title) < 10:
                continue

            # Quick exact match
            if title_lower == seen_title:
                return True

            # Fuzzy match for similar titles
            similarity = fuzz.ratio(title_lower, seen_title)
            if similarity >= self.title_similarity_threshold:
                return True

            # Token-based similarity (handles word reordering)
            token_similarity = fuzz.token_sort_ratio(title_lower, seen_title)
            if token_similarity >= 95:
                return True

        return False

    def merge_duplicates(
        self,
        evidence_list: List[Dict],
        merge_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Merge duplicate entries instead of discarding them.

        Useful when duplicates might have different metadata
        (e.g., different snippets from different queries).

        Args:
            evidence_list: List of evidence dictionaries
            merge_fields: Fields to merge (default: sources, queries)

        Returns:
            Deduplicated list with merged metadata
        """
        if not evidence_list:
            return []

        merge_fields = merge_fields or ["sources", "queries", "search_queries"]

        # Group by canonical URL
        url_groups: Dict[str, List[Dict]] = {}

        for item in evidence_list:
            url = item.get("url", "")
            if not url:
                continue

            canonical = self._canonicalize_url(url)
            if canonical not in url_groups:
                url_groups[canonical] = []
            url_groups[canonical].append(item)

        # Merge groups
        merged_results: List[Dict] = []

        for canonical_url, group in url_groups.items():
            if len(group) == 1:
                merged_results.append(group[0])
            else:
                # Merge the group
                merged = self._merge_group(group, merge_fields)
                merged_results.append(merged)

        logger.info(
            f"Merged {len(evidence_list)} items into {len(merged_results)} unique entries"
        )

        return merged_results

    def _merge_group(
        self,
        group: List[Dict],
        merge_fields: List[str],
    ) -> Dict:
        """
        Merge a group of duplicate entries.

        Args:
            group: List of duplicate entries
            merge_fields: Fields to combine

        Returns:
            Single merged entry
        """
        # Start with the first item as base
        merged = dict(group[0])

        # Track merged field values
        for field in merge_fields:
            values = set()
            for item in group:
                value = item.get(field)
                if value:
                    if isinstance(value, list):
                        values.update(value)
                    else:
                        values.add(value)
            if values:
                merged[field] = list(values)

        # Keep the longest snippet
        longest_snippet = ""
        for item in group:
            snippet = item.get("snippet", item.get("content", ""))
            if len(snippet) > len(longest_snippet):
                longest_snippet = snippet

        if longest_snippet:
            merged["snippet"] = longest_snippet

        # Track duplicate count
        merged["duplicate_count"] = len(group)

        return merged

    def find_near_duplicates(
        self,
        evidence_list: List[Dict],
        threshold: int = 80,
    ) -> List[List[int]]:
        """
        Find groups of near-duplicate entries (for manual review).

        Args:
            evidence_list: List of evidence dictionaries
            threshold: Similarity threshold (0-100)

        Returns:
            List of groups, where each group is a list of indices
        """
        if not evidence_list:
            return []

        n = len(evidence_list)
        groups: List[Set[int]] = []
        assigned: Set[int] = set()

        for i in range(n):
            if i in assigned:
                continue

            group = {i}
            item_i = evidence_list[i]
            url_i = self._canonicalize_url(item_i.get("url", ""))
            title_i = item_i.get("title", "").lower()

            for j in range(i + 1, n):
                if j in assigned:
                    continue

                item_j = evidence_list[j]
                url_j = self._canonicalize_url(item_j.get("url", ""))
                title_j = item_j.get("title", "").lower()

                # Check URL similarity
                url_sim = fuzz.ratio(url_i, url_j) if url_i and url_j else 0

                # Check title similarity
                title_sim = fuzz.ratio(title_i, title_j) if title_i and title_j else 0

                # If either is above threshold, they're near-duplicates
                if url_sim >= threshold or title_sim >= threshold:
                    group.add(j)

            if len(group) > 1:
                groups.append(group)
                assigned.update(group)

        return [list(g) for g in groups]


def deduplicate_evidence(
    evidence_list: List[Dict],
    url_threshold: int = 90,
    content_threshold: int = 85,
) -> List[Dict]:
    """
    Convenience function for deduplication.

    Args:
        evidence_list: List of evidence dictionaries
        url_threshold: URL similarity threshold (0-100)
        content_threshold: Content similarity threshold (0-100)

    Returns:
        Deduplicated list
    """
    deduper = Deduplicator(
        url_similarity_threshold=url_threshold,
        content_similarity_threshold=content_threshold,
    )
    return deduper.deduplicate(evidence_list)


def merge_evidence(
    evidence_list: List[Dict],
    merge_fields: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Merge duplicate evidence entries, combining metadata.

    Args:
        evidence_list: List of evidence dictionaries
        merge_fields: Fields to merge across duplicates

    Returns:
        Merged evidence list
    """
    deduper = Deduplicator()
    return deduper.merge_duplicates(evidence_list, merge_fields)
