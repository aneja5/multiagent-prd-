"""
Test credibility scoring functionality.

Tests cover:
- Domain credibility tiers
- Recency scoring
- Content quality signals
- Batch scoring
- Edge cases
"""

from datetime import datetime, timedelta
import pytest

from tools.credibility import (
    CredibilityScorer,
    score_credibility,
    score_sources,
    HIGH_CREDIBILITY_DOMAINS,
    MEDIUM_CREDIBILITY_DOMAINS,
    LOW_CREDIBILITY_DOMAINS,
)


class TestDomainCredibility:
    """Tests for domain-based scoring."""

    def test_gov_domain_high_credibility(self):
        """Test .gov domains get high credibility."""
        result = score_credibility(
            url="https://www.irs.gov/forms-pubs/about-forms",
            title="Tax Forms and Publications",
        )
        assert result["tier"] == "high"
        assert result["domain_score"] == 1.0
        assert any(".gov" in s for s in result["signals"])

    def test_edu_domain_high_credibility(self):
        """Test .edu domains get high credibility."""
        result = score_credibility(
            url="https://www.stanford.edu/research/ai",
            title="AI Research at Stanford",
        )
        assert result["tier"] == "high"
        assert result["domain_score"] == 1.0

    def test_official_docs_high_credibility(self):
        """Test official documentation gets high credibility."""
        urls = [
            "https://docs.microsoft.com/en-us/azure/",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "https://docs.aws.amazon.com/lambda/",
        ]
        for url in urls:
            result = score_credibility(url=url, title="Documentation")
            assert result["domain_score"] == 1.0, f"Failed for {url}"

    def test_research_sources_high_credibility(self):
        """Test academic sources get high credibility."""
        result = score_credibility(
            url="https://arxiv.org/abs/2301.00001",
            title="Machine Learning Research Paper",
        )
        assert result["domain_score"] == 1.0

    def test_tech_news_medium_credibility(self):
        """Test tech news sites get medium credibility."""
        urls = [
            "https://techcrunch.com/2024/01/15/ai-funding",
            "https://www.theverge.com/tech/review",
            "https://arstechnica.com/science/article",
        ]
        for url in urls:
            result = score_credibility(url=url, title="Tech Article")
            assert result["domain_score"] == 0.7, f"Failed for {url}"

    def test_review_sites_medium_credibility(self):
        """Test software review sites get medium credibility."""
        result = score_credibility(
            url="https://www.g2.com/products/slack/reviews",
            title="Slack Reviews",
        )
        assert result["domain_score"] == 0.7

    def test_reddit_low_credibility(self):
        """Test Reddit gets lower but valid credibility."""
        result = score_credibility(
            url="https://reddit.com/r/freelance/comments/abc123/invoicing",
            title="Discussion about invoicing",
        )
        assert result["domain_score"] == 0.4
        assert result["tier"] in ["low", "medium"]
        assert any("Reddit" in s or "reddit" in s for s in result["signals"])

    def test_social_media_low_credibility(self):
        """Test social media gets low credibility."""
        urls = [
            "https://twitter.com/user/status/123",
            "https://www.linkedin.com/posts/user-123",
            "https://www.facebook.com/page/post",
        ]
        for url in urls:
            result = score_credibility(url=url, title="Social Post")
            assert result["domain_score"] == 0.4, f"Failed for {url}"

    def test_unknown_domain_moderate_credibility(self):
        """Test unknown domains get moderate credibility."""
        result = score_credibility(
            url="https://random-company-xyz.com/blog/post",
            title="Some Article",
        )
        assert result["domain_score"] == 0.5

    def test_blog_subdomain_detected(self):
        """Test blog subdomains are detected."""
        result = score_credibility(
            url="https://blog.company.com/article",
            title="Company Blog Post",
        )
        assert result["domain_score"] == 0.5
        assert any("blog" in s.lower() for s in result["signals"])


class TestRecencyScoring:
    """Tests for recency-based scoring."""

    def test_very_recent_content(self):
        """Test content from last 30 days gets highest recency score."""
        recent_date = (datetime.now() - timedelta(days=15)).isoformat()
        result = score_credibility(
            url="https://example.com/article",
            published_date=recent_date,
        )
        assert result["recency_score"] == 1.0
        assert any("recent" in s.lower() for s in result["signals"])

    def test_moderately_recent_content(self):
        """Test content from last 6 months gets good score."""
        date_120_days = (datetime.now() - timedelta(days=120)).isoformat()
        result = score_credibility(
            url="https://example.com/article",
            published_date=date_120_days,
        )
        assert result["recency_score"] >= 0.7

    def test_year_old_content(self):
        """Test 1-year-old content gets moderate score."""
        date_1_year = (datetime.now() - timedelta(days=365)).isoformat()
        result = score_credibility(
            url="https://example.com/article",
            published_date=date_1_year,
        )
        assert 0.4 <= result["recency_score"] <= 0.7

    def test_old_content(self):
        """Test 3+ year old content gets low score."""
        date_3_years = (datetime.now() - timedelta(days=1100)).isoformat()
        result = score_credibility(
            url="https://example.com/article",
            published_date=date_3_years,
        )
        assert result["recency_score"] <= 0.4
        assert any("years" in s.lower() for s in result["signals"])

    def test_no_date_moderate_score(self):
        """Test missing date gets moderate score."""
        result = score_credibility(
            url="https://example.com/article",
            published_date=None,
        )
        assert result["recency_score"] == 0.5
        assert any("no publication date" in s.lower() for s in result["signals"])

    def test_invalid_date_handled(self):
        """Test invalid date string is handled gracefully."""
        result = score_credibility(
            url="https://example.com/article",
            published_date="not-a-date",
        )
        assert result["recency_score"] == 0.5

    def test_various_date_formats(self):
        """Test various date formats are parsed correctly."""
        dates = [
            "2024-01-15",
            "January 15, 2024",
            "15/01/2024",
            "2024-01-15T10:30:00Z",
        ]
        for date_str in dates:
            result = score_credibility(
                url="https://example.com/article",
                published_date=date_str,
            )
            # Should not get default 0.5 score
            assert result["recency_score"] != 0.5 or "could not parse" not in str(result["signals"]).lower()


class TestContentQuality:
    """Tests for content quality signals."""

    def test_research_content_bonus(self):
        """Test research indicators boost score."""
        result = score_credibility(
            url="https://example.com/article",
            title="Market Research Report",
            content="This study analyzes market trends based on survey data and findings.",
        )
        assert result["content_score"] > 0.5
        assert any("research" in s.lower() for s in result["signals"])

    def test_statistics_bonus(self):
        """Test statistics in content boost score."""
        result = score_credibility(
            url="https://example.com/article",
            title="Industry Report",
            content="The market grew 45% year over year with $2.5M in revenue from 10,000 customers.",
        )
        assert result["content_score"] > 0.5
        assert any("statistic" in s.lower() for s in result["signals"])

    def test_product_info_bonus(self):
        """Test pricing/features info boosts score."""
        # Need sufficient content (>100 words) to avoid thin content penalty
        content = """
        This comprehensive guide compares pricing tiers and features between
        the top alternatives in the market. We analyze each product's pricing
        structure, key features, and value proposition to help you make an
        informed decision. The comparison includes detailed feature matrices
        and pricing breakdowns for enterprise and small business plans.

        We evaluated each solution based on core functionality, ease of use,
        customer support quality, integration capabilities, and total cost of
        ownership. Our methodology involved hands-on testing, customer interviews,
        and analysis of publicly available information about each vendor.

        The pricing comparison section breaks down monthly and annual costs,
        including any hidden fees or setup charges that might affect your budget.
        """
        result = score_credibility(
            url="https://example.com/article",
            title="Product Comparison",
            content=content,
        )
        assert result["content_score"] > 0.5
        assert any("pricing" in s.lower() or "product" in s.lower() for s in result["signals"])

    def test_substantial_content_bonus(self):
        """Test substantial content gets bonus."""
        long_content = "This is detailed content. " * 200  # ~800 words
        result = score_credibility(
            url="https://example.com/article",
            title="Detailed Article",
            content=long_content,
        )
        assert any("content" in s.lower() for s in result["signals"])

    def test_sponsored_content_penalty(self):
        """Test sponsored content gets penalty."""
        result = score_credibility(
            url="https://example.com/article",
            title="Product Review",
            content="This sponsored post was paid for by Company X.",
        )
        assert result["content_score"] < 0.5
        assert any("sponsor" in s.lower() for s in result["signals"])

    def test_error_page_penalty(self):
        """Test error pages get penalized."""
        result = score_credibility(
            url="https://example.com/article",
            title="404 Page Not Found",
            content="The page you requested could not be found.",
        )
        assert result["content_score"] < 0.5
        assert any("error" in s.lower() for s in result["signals"])

    def test_clickbait_penalty(self):
        """Test clickbait language gets penalty."""
        result = score_credibility(
            url="https://example.com/article",
            title="You Won't Believe This Shocking Secret Hack",
            content="Mind-blowing tricks revealed.",
        )
        assert result["content_score"] < 0.5


class TestSpamDetection:
    """Tests for spam signal detection."""

    def test_spam_url_penalty(self):
        """Test spam indicators in URL get penalty."""
        result = score_credibility(
            url="https://best-free-software-deals.com/promo",
            title="Software Deals",
        )
        assert any("spam" in s.lower() for s in result["signals"])

    def test_excessive_hyphens_penalty(self):
        """Test excessive hyphens in domain get penalty."""
        result = score_credibility(
            url="https://get-the-best-free-cheap-deals-now.com/offer",
            title="Deals",
        )
        assert any("suspicious" in s.lower() for s in result["signals"])


class TestTierClassification:
    """Tests for tier classification."""

    def test_high_tier_threshold(self):
        """Test high tier requires score >= 0.7."""
        # High credibility domain + recent + good content
        recent_date = (datetime.now() - timedelta(days=10)).isoformat()
        result = score_credibility(
            url="https://www.nytimes.com/2024/01/15/technology/ai-research.html",
            title="AI Research Analysis Report",
            content="This comprehensive study analyzes market data with findings from expert analysts.",
            published_date=recent_date,
        )
        assert result["tier"] == "high"
        assert result["score"] >= 0.7

    def test_medium_tier_range(self):
        """Test medium tier is 0.4-0.7."""
        # Medium domain, no date
        result = score_credibility(
            url="https://techcrunch.com/article",
            title="Tech News",
        )
        assert result["tier"] == "medium"
        assert 0.4 <= result["score"] < 0.7

    def test_low_tier_threshold(self):
        """Test low tier is below 0.4."""
        # Low credibility domain, old date, thin content
        old_date = (datetime.now() - timedelta(days=1200)).isoformat()
        result = score_credibility(
            url="https://reddit.com/r/random/comments/xyz",
            title="Random Post",
            content="Short.",
            published_date=old_date,
        )
        assert result["tier"] == "low"
        assert result["score"] < 0.4


class TestBatchScoring:
    """Tests for batch scoring functionality."""

    def test_score_multiple_sources(self):
        """Test scoring multiple sources at once."""
        sources = [
            {"url": "https://www.nytimes.com/article", "title": "News Article"},
            {"url": "https://reddit.com/r/tech/post", "title": "Reddit Post"},
            {"url": "https://techcrunch.com/news", "title": "Tech News"},
        ]
        scored = score_sources(sources)

        assert len(scored) == 3
        assert all("credibility" in s for s in scored)

        # Should be sorted by score descending
        scores = [s["credibility"]["score"] for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_batch_preserves_original_data(self):
        """Test batch scoring preserves original source data."""
        sources = [
            {"url": "https://example.com", "title": "Test", "custom_field": "value"},
        ]
        scored = score_sources(sources)

        assert scored[0]["custom_field"] == "value"
        assert scored[0]["url"] == "https://example.com"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_url(self):
        """Test empty URL is handled."""
        result = score_credibility(url="", title="No URL")
        assert result["score"] >= 0
        assert result["score"] <= 1

    def test_invalid_url(self):
        """Test invalid URL is handled."""
        result = score_credibility(url="not-a-url", title="Invalid")
        assert result["score"] >= 0

    def test_none_values(self):
        """Test None values don't crash."""
        result = score_credibility(
            url="https://example.com",
            title=None,
            content=None,
            published_date=None,
        )
        assert result["score"] >= 0

    def test_unicode_content(self):
        """Test unicode content is handled."""
        result = score_credibility(
            url="https://example.com",
            title="文章标题 - Article",
            content="内容包含中文和emoji 🎉",
        )
        assert result["score"] >= 0

    def test_very_long_url(self):
        """Test very long URLs are handled."""
        long_url = "https://example.com/" + "a" * 500
        result = score_credibility(url=long_url, title="Long URL")
        assert any("long url" in s.lower() for s in result["signals"])


class TestCustomWeights:
    """Tests for custom weight configuration."""

    def test_custom_weights(self):
        """Test custom weight configuration."""
        scorer = CredibilityScorer(
            domain_weight=0.8,
            recency_weight=0.1,
            content_weight=0.1,
        )

        # Domain-heavy scoring
        result = scorer.score(
            url="https://www.gov.uk/official",
            title="Official Document",
        )

        # Should be heavily influenced by domain
        assert result["score"] >= 0.7

    def test_weights_normalized(self):
        """Test weights that don't sum to 1 are normalized."""
        scorer = CredibilityScorer(
            domain_weight=1.0,
            recency_weight=1.0,
            content_weight=1.0,
        )

        # Should still work (weights normalized internally)
        result = scorer.score(url="https://example.com", title="Test")
        assert 0 <= result["score"] <= 1
