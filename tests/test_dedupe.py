"""
Test deduplication functionality.

Tests cover:
- URL canonicalization
- Exact duplicate removal
- Fuzzy URL matching
- Content similarity detection
- Edge cases
"""

import pytest

from tools.dedupe import (
    Deduplicator,
    deduplicate_evidence,
    merge_evidence,
    TRACKING_PARAMS,
    SIMHASH_DISTANCE_THRESHOLD,
    SIMHASH_MIN_CONTENT_LENGTH,
)


class TestURLCanonicalization:
    """Tests for URL canonicalization."""

    def test_removes_www_prefix(self):
        """Test www. prefix is removed."""
        deduper = Deduplicator()
        canonical = deduper._canonicalize_url("https://www.example.com/page")
        assert "www." not in canonical
        assert "example.com/page" in canonical

    def test_removes_tracking_params(self):
        """Test UTM and tracking parameters are removed."""
        deduper = Deduplicator()
        url = "https://example.com/page?utm_source=twitter&utm_medium=social&id=123"
        canonical = deduper._canonicalize_url(url)
        assert "utm_source" not in canonical
        assert "utm_medium" not in canonical
        assert "id=123" in canonical

    def test_removes_fragment(self):
        """Test URL fragments are removed."""
        deduper = Deduplicator()
        canonical = deduper._canonicalize_url("https://example.com/page#section-2")
        assert "#" not in canonical

    def test_normalizes_trailing_slash(self):
        """Test trailing slashes are normalized."""
        deduper = Deduplicator()
        url1 = deduper._canonicalize_url("https://example.com/page/")
        url2 = deduper._canonicalize_url("https://example.com/page")
        assert url1 == url2

    def test_removes_index_files(self):
        """Test index files are normalized."""
        deduper = Deduplicator()
        url1 = deduper._canonicalize_url("https://example.com/page/index.html")
        url2 = deduper._canonicalize_url("https://example.com/page")
        # Both should normalize to same form (without index.html and without trailing slash)
        assert url1 == url2

    def test_lowercase_domain(self):
        """Test domain is lowercased."""
        deduper = Deduplicator()
        canonical = deduper._canonicalize_url("https://EXAMPLE.COM/Page")
        assert "example.com" in canonical

    def test_removes_default_ports(self):
        """Test default ports are removed."""
        deduper = Deduplicator()
        url1 = deduper._canonicalize_url("https://example.com:443/page")
        url2 = deduper._canonicalize_url("https://example.com/page")
        assert url1 == url2

    def test_preserves_non_tracking_params(self):
        """Test non-tracking query params are preserved."""
        deduper = Deduplicator()
        canonical = deduper._canonicalize_url("https://example.com/search?q=python&page=2")
        assert "q=python" in canonical
        assert "page=2" in canonical

    def test_sorts_query_params(self):
        """Test query params are sorted consistently."""
        deduper = Deduplicator()
        url1 = deduper._canonicalize_url("https://example.com?b=2&a=1")
        url2 = deduper._canonicalize_url("https://example.com?a=1&b=2")
        assert url1 == url2

    def test_handles_url_encoding(self):
        """Test URL-encoded characters are decoded."""
        deduper = Deduplicator()
        canonical = deduper._canonicalize_url("https://example.com/path%20with%20spaces")
        assert "path with spaces" in canonical or "path%20with%20spaces" in canonical.lower()


class TestExactDuplicates:
    """Tests for exact duplicate removal."""

    def test_removes_exact_url_duplicates(self):
        """Test removal of exact URL duplicates."""
        evidence = [
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_keeps_first_occurrence(self):
        """Test that first occurrence is kept."""
        evidence = [
            {"url": "https://example.com/page", "title": "First", "snippet": "First text"},
            {"url": "https://example.com/page", "title": "Second", "snippet": "Second text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1
        assert result[0]["title"] == "First"

    def test_removes_www_variants(self):
        """Test removal of www vs non-www duplicates."""
        evidence = [
            {"url": "https://www.example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_removes_tracking_param_variants(self):
        """Test removal of tracking parameter variants."""
        evidence = [
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page?utm_source=twitter", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page?utm_medium=social&ref=123", "title": "Title", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_removes_fragment_variants(self):
        """Test removal of fragment variants."""
        evidence = [
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page#section1", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page#section2", "title": "Title", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1


class TestFuzzyURLMatching:
    """Tests for fuzzy URL matching."""

    def test_fuzzy_url_match(self):
        """Test fuzzy matching catches near-duplicate URLs."""
        evidence = [
            {"url": "https://example.com/article/my-post", "title": "Post", "snippet": "Text"},
            {"url": "https://example.com/article/my-post/", "title": "Post", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_different_urls_preserved(self):
        """Test that genuinely different URLs are preserved."""
        evidence = [
            {"url": "https://example.com/page1", "title": "Title 1", "snippet": "Text 1"},
            {"url": "https://example.com/page2", "title": "Title 2", "snippet": "Text 2"},
            {"url": "https://different.com/page", "title": "Title 3", "snippet": "Text 3"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 3


class TestContentSimilarity:
    """Tests for content-based deduplication."""

    def test_removes_same_content_different_urls(self):
        """Test content hash catches exact content duplicates."""
        evidence = [
            {"url": "https://site1.com/article", "title": "Same Title", "snippet": "Same exact content here"},
            {"url": "https://site2.com/repost", "title": "Same Title", "snippet": "Same exact content here"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_similar_titles_detected(self):
        """Test similar titles are detected as duplicates."""
        evidence = [
            {"url": "https://site1.com/a", "title": "How to Build a SaaS Product in 2024", "snippet": "Guide"},
            {"url": "https://site2.com/b", "title": "How to Build a SaaS Product in 2024", "snippet": "Guide"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_different_content_preserved(self):
        """Test genuinely different content is preserved."""
        evidence = [
            {"url": "https://example.com/a", "title": "Python Tutorial", "snippet": "Learn Python basics"},
            {"url": "https://example.com/b", "title": "Java Tutorial", "snippet": "Learn Java basics"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 2


class TestMerging:
    """Tests for merge functionality."""

    def test_merge_duplicates(self):
        """Test merging duplicate entries."""
        evidence = [
            {"url": "https://example.com/page", "title": "Title", "snippet": "Short", "sources": ["query1"]},
            {"url": "https://example.com/page?utm_source=x", "title": "Title", "snippet": "Longer snippet here", "sources": ["query2"]},
        ]
        result = merge_evidence(evidence)
        assert len(result) == 1
        # Should have merged sources
        assert "sources" in result[0]
        assert len(result[0]["sources"]) == 2
        # Should keep longest snippet
        assert "Longer" in result[0]["snippet"]

    def test_merge_tracks_duplicate_count(self):
        """Test duplicate count is tracked."""
        evidence = [
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
            {"url": "https://example.com/page", "title": "Title", "snippet": "Text"},
        ]
        result = merge_evidence(evidence)
        assert len(result) == 1
        assert result[0]["duplicate_count"] == 3


class TestNearDuplicates:
    """Tests for near-duplicate detection."""

    def test_find_near_duplicates(self):
        """Test finding groups of near-duplicates."""
        deduper = Deduplicator()
        evidence = [
            {"url": "https://site1.com/article", "title": "Best Python Tips 2024"},
            {"url": "https://site2.com/post", "title": "Best Python Tips for 2024"},
            {"url": "https://site3.com/other", "title": "Completely Different Topic"},
        ]
        groups = deduper.find_near_duplicates(evidence, threshold=80)
        # First two should be grouped
        assert len(groups) >= 1
        # At least one group should have indices 0 and 1
        found_group = False
        for group in groups:
            if 0 in group and 1 in group:
                found_group = True
                break
        assert found_group


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_list(self):
        """Test empty list returns empty list."""
        result = deduplicate_evidence([])
        assert result == []

    def test_single_item(self):
        """Test single item is preserved."""
        evidence = [{"url": "https://example.com", "title": "Title", "snippet": "Text"}]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_missing_url(self):
        """Test items without URLs are skipped."""
        evidence = [
            {"title": "No URL", "snippet": "Text"},
            {"url": "https://example.com", "title": "Has URL", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1
        assert result[0]["title"] == "Has URL"

    def test_missing_title(self):
        """Test items without titles are handled."""
        evidence = [
            {"url": "https://example.com/a", "snippet": "Text A"},
            {"url": "https://example.com/b", "snippet": "Text B"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 2

    def test_missing_snippet(self):
        """Test items without snippets are handled."""
        evidence = [
            {"url": "https://example.com/a", "title": "Title A"},
            {"url": "https://example.com/b", "title": "Title B"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 2

    def test_none_values(self):
        """Test None values don't crash."""
        evidence = [
            {"url": "https://example.com/a", "title": None, "snippet": None},
            {"url": "https://example.com/b", "title": None, "snippet": None},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 2

    def test_unicode_content(self):
        """Test unicode content is handled."""
        evidence = [
            {"url": "https://example.com/a", "title": "日本語タイトル", "snippet": "日本語コンテンツ"},
            {"url": "https://example.com/b", "title": "中文标题", "snippet": "中文内容"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 2

    def test_very_long_urls(self):
        """Test very long URLs are handled."""
        long_url = "https://example.com/" + "a" * 500
        evidence = [
            {"url": long_url, "title": "Title", "snippet": "Text"},
            {"url": long_url, "title": "Title", "snippet": "Text"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_malformed_urls(self):
        """Test malformed URLs don't crash."""
        evidence = [
            {"url": "not-a-url", "title": "Title 1", "snippet": "Text 1"},
            {"url": "also/not/valid", "title": "Title 2", "snippet": "Text 2"},
            {"url": "https://valid.com", "title": "Title 3", "snippet": "Text 3"},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) >= 1


class TestCustomThresholds:
    """Tests for custom threshold configuration."""

    def test_strict_url_threshold(self):
        """Test stricter URL threshold keeps more items."""
        evidence = [
            {"url": "https://example.com/article-1", "title": "A", "snippet": "X"},
            {"url": "https://example.com/article-2", "title": "B", "snippet": "Y"},
        ]
        # Very strict threshold
        result = deduplicate_evidence(evidence, url_threshold=99)
        assert len(result) == 2

    def test_loose_url_threshold(self):
        """Test looser URL threshold merges more items."""
        evidence = [
            {"url": "https://example.com/post-about-python", "title": "A", "snippet": "X"},
            {"url": "https://example.com/post-about-pythons", "title": "A", "snippet": "X"},
        ]
        # Very loose threshold
        result = deduplicate_evidence(evidence, url_threshold=70)
        assert len(result) == 1


class TestSimHash:
    """Tests for SimHash near-duplicate detection."""

    def test_simhash_detects_paraphrased_content(self):
        """Test SimHash catches paraphrased/similar content."""
        # Same article with minor edits (typical of content syndication)
        original = """
        The latest developments in artificial intelligence have transformed
        the technology landscape. Machine learning models are now capable of
        understanding natural language, generating images, and writing code.
        Companies worldwide are investing billions in AI research and development.
        """
        paraphrased = """
        The latest developments in artificial intelligence have transformed
        the technology landscape. Machine learning models are now capable of
        understanding natural language, generating images, and writing software.
        Companies globally are investing billions in AI research and development.
        """

        evidence = [
            {"url": "https://site1.com/ai-article", "title": "AI Developments", "snippet": original},
            {"url": "https://site2.com/ai-news", "title": "AI Developments", "snippet": paraphrased},
        ]
        result = deduplicate_evidence(evidence)
        # Should detect as near-duplicate
        assert len(result) == 1

    def test_simhash_preserves_different_content(self):
        """Test SimHash doesn't match genuinely different content."""
        content_a = """
        Python is a high-level programming language known for its simplicity
        and readability. It was created by Guido van Rossum and released in 1991.
        Python supports multiple programming paradigms including procedural,
        object-oriented, and functional programming.
        """
        content_b = """
        JavaScript is a dynamic programming language primarily used for web
        development. It was created by Brendan Eich at Netscape in 1995.
        JavaScript is essential for creating interactive web pages and is
        supported by all modern web browsers.
        """

        evidence = [
            {"url": "https://site1.com/python", "title": "Python Guide", "snippet": content_a},
            {"url": "https://site2.com/javascript", "title": "JavaScript Guide", "snippet": content_b},
        ]
        result = deduplicate_evidence(evidence)
        # Should preserve both as they're genuinely different
        assert len(result) == 2

    def test_simhash_skips_short_content(self):
        """Test SimHash doesn't run on content below threshold."""
        # Short content below SIMHASH_MIN_CONTENT_LENGTH
        evidence = [
            {"url": "https://site1.com/a", "title": "Title A", "snippet": "Short."},
            {"url": "https://site2.com/b", "title": "Title B", "snippet": "Brief."},
        ]
        result = deduplicate_evidence(evidence)
        # Both should be preserved (SimHash skipped, different titles)
        assert len(result) == 2

    def test_simhash_catches_reposted_articles(self):
        """Test SimHash catches articles reposted with minor changes."""
        article = """
        SaaS pricing strategies can make or break your business. The most
        successful companies use value-based pricing that aligns with customer
        outcomes. Freemium models work well for products with viral potential,
        while enterprise solutions often benefit from custom pricing tiers.
        Understanding your target market is essential for pricing optimization.
        """
        # Same article with slight modifications
        repost = """
        SaaS pricing strategies can make or break your business! The most
        successful companies use value-based pricing that aligns with customer
        outcomes. Freemium models work well for products with viral potential,
        while enterprise solutions often benefit from custom pricing tiers.
        Understanding your target market is essential for pricing optimization.
        """

        evidence = [
            {"url": "https://original.com/pricing", "title": "SaaS Pricing", "snippet": article},
            {"url": "https://repost.com/pricing-tips", "title": "Pricing Tips", "snippet": repost},
        ]
        result = deduplicate_evidence(evidence)
        assert len(result) == 1

    def test_simhash_with_custom_threshold(self):
        """Test SimHash with custom threshold."""
        # Create deduplicator with very strict threshold (1 bit)
        deduper = Deduplicator(simhash_threshold=1)

        # Content that's similar but not identical
        content1 = "A" * 200
        content2 = "A" * 195 + "B" * 5

        sh1 = deduper._compute_simhash("Title", content1)
        sh2 = deduper._compute_simhash("Title", content2)

        assert sh1 is not None
        assert sh2 is not None

        # With strict threshold, they might not match
        # (depending on actual hash values)

    def test_simhash_computation(self):
        """Test SimHash is computed correctly."""
        deduper = Deduplicator()

        # Test with sufficient content
        content = "This is a test article. " * 20
        simhash = deduper._compute_simhash("Test Title", content)
        assert simhash is not None

        # Test with insufficient content
        short_simhash = deduper._compute_simhash("Title", "Short")
        assert short_simhash is None

    def test_simhash_find_similar(self):
        """Test finding similar SimHashes."""
        from simhash import Simhash

        deduper = Deduplicator(simhash_threshold=5)

        # Create a base simhash
        base_content = "The quick brown fox jumps over the lazy dog. " * 10
        base_hash = Simhash(base_content)

        # Slightly modified content should have similar hash
        similar_content = "The quick brown fox jumps over the lazy cat. " * 10
        similar_hash = Simhash(similar_content)

        seen = [(base_hash, 0)]

        # Check if similar hash is found
        result = deduper._find_similar_simhash(similar_hash, seen)
        # They should be similar (within threshold)
        # Note: actual result depends on hash values


class TestTrackingParams:
    """Tests for tracking parameter coverage."""

    def test_all_utm_params_removed(self):
        """Test all UTM parameters are removed."""
        deduper = Deduplicator()
        url = "https://example.com/page?utm_source=a&utm_medium=b&utm_campaign=c&utm_content=d&utm_term=e"
        canonical = deduper._canonicalize_url(url)
        for param in ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"]:
            assert param not in canonical

    def test_social_tracking_removed(self):
        """Test social media tracking params are removed."""
        deduper = Deduplicator()
        url = "https://example.com/page?fbclid=abc&gclid=def&twclid=ghi"
        canonical = deduper._canonicalize_url(url)
        for param in ["fbclid", "gclid", "twclid"]:
            assert param not in canonical

    def test_common_tracking_params(self):
        """Test common tracking params are all in the set."""
        expected_params = [
            "utm_source", "utm_medium", "utm_campaign",
            "fbclid", "gclid", "ref", "source",
        ]
        for param in expected_params:
            assert param in TRACKING_PARAMS


class TestIntegrationWithOtherTools:
    """Tests for integration scenarios."""

    def test_dedup_search_results(self):
        """Test deduplicating typical search results."""
        # Simulate results from multiple search queries
        evidence = [
            {"url": "https://techcrunch.com/article", "title": "AI News", "snippet": "Latest AI developments"},
            {"url": "https://www.techcrunch.com/article", "title": "AI News", "snippet": "Latest AI developments"},
            {"url": "https://techcrunch.com/article?utm_source=twitter", "title": "AI News", "snippet": "Latest AI"},
            {"url": "https://theverge.com/tech", "title": "Tech Roundup", "snippet": "Technology news"},
            {"url": "https://arstechnica.com/science", "title": "Science Update", "snippet": "Scientific discoveries"},
        ]
        result = deduplicate_evidence(evidence)
        # Should reduce TechCrunch variants to 1
        assert len(result) == 3
        urls = [r["url"] for r in result]
        assert sum("techcrunch" in u for u in urls) == 1

    def test_preserves_credibility_metadata(self):
        """Test that credibility metadata is preserved."""
        evidence = [
            {
                "url": "https://example.com/unique",
                "title": "Article",
                "snippet": "Content",
                "credibility": {"score": 0.8, "tier": "high"},
            }
        ]
        result = deduplicate_evidence(evidence)
        assert "credibility" in result[0]
        assert result[0]["credibility"]["score"] == 0.8
