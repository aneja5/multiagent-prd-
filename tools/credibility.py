"""
Score source credibility based on domain, content type, and signals.

This module provides credibility scoring for research sources to help
prioritize high-quality evidence and flag weak claims.
"""

from typing import Dict, List, Optional, Set
from urllib.parse import urlparse
from datetime import datetime, timezone
import re

from dateutil import parser as date_parser

from app.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Domain Credibility Tiers
# =============================================================================

# High credibility: Official, peer-reviewed, established institutions
HIGH_CREDIBILITY_DOMAINS: Set[str] = {
    # Government domains (TLDs)
    ".gov",
    ".gov.uk",
    ".gov.au",
    ".gov.ca",
    ".mil",

    # Education domains (TLDs)
    ".edu",
    ".ac.uk",
    ".edu.au",

    # Official documentation
    "docs.microsoft.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "docs.aws.amazon.com",
    "cloud.google.com/docs",
    "docs.github.com",
    "docs.python.org",
    "docs.oracle.com",
    "developer.apple.com",
    "developers.google.com",

    # Established news & media
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "economist.com",
    "ft.com",
    "apnews.com",
    "npr.org",
    "theguardian.com",

    # Industry research & analysis
    "gartner.com",
    "forrester.com",
    "idc.com",
    "mckinsey.com",
    "hbr.org",
    "statista.com",
    "pewresearch.org",

    # Academic & peer-reviewed
    "arxiv.org",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "ieee.org",
    "acm.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "jstor.org",
    "scholar.google.com",

    # Standards organizations
    "w3.org",
    "ietf.org",
    "iso.org",
    "nist.gov",
}

# Medium credibility: Reputable but editorial/commercial
MEDIUM_CREDIBILITY_DOMAINS: Set[str] = {
    # Tech news & analysis
    "techcrunch.com",
    "theverge.com",
    "arstechnica.com",
    "wired.com",
    "zdnet.com",
    "cnet.com",
    "engadget.com",
    "venturebeat.com",
    "theregister.com",
    "infoworld.com",
    "computerworld.com",

    # Business news
    "forbes.com",
    "businessinsider.com",
    "inc.com",
    "entrepreneur.com",
    "fastcompany.com",
    "fortune.com",
    "cnbc.com",

    # Software review platforms
    "g2.com",
    "capterra.com",
    "trustpilot.com",
    "producthunt.com",
    "softwareadvice.com",
    "getapp.com",
    "trustradius.com",

    # Developer resources
    "stackoverflow.com",
    "github.com",
    "gitlab.com",
    "dev.to",
    "hackernoon.com",
    "dzone.com",
    "infoq.com",
    "smashingmagazine.com",
    "css-tricks.com",

    # Publishing platforms (quality varies)
    "medium.com",
    "substack.com",
    "hashnode.dev",

    # Wikipedia (generally reliable with citations)
    "wikipedia.org",
    "en.wikipedia.org",
}

# Low credibility: User-generated, social (valuable but not authoritative)
LOW_CREDIBILITY_DOMAINS: Set[str] = {
    # Forums & Q&A
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "discourse.org",

    # Social media
    "twitter.com",
    "x.com",
    "facebook.com",
    "linkedin.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",

    # Personal blogs (no editorial review)
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",

    # Content farms & aggregators
    "buzzfeed.com",
    "huffpost.com",
}

# Known spam/low-quality indicators
SPAM_INDICATORS: Set[str] = {
    "free-",
    "-free",
    "cheap-",
    "-cheap",
    "best-",
    "-best",
    "click",
    "affiliate",
    "promo",
    "deal",
}


class CredibilityScorer:
    """
    Score source credibility on 0-1 scale.

    Uses multiple signals including domain reputation, content recency,
    and quality indicators to produce a weighted credibility score.
    """

    def __init__(
        self,
        domain_weight: float = 0.5,
        recency_weight: float = 0.3,
        content_weight: float = 0.2,
    ):
        """
        Initialize the credibility scorer.

        Args:
            domain_weight: Weight for domain reputation (default: 0.5)
            recency_weight: Weight for content recency (default: 0.3)
            content_weight: Weight for content quality (default: 0.2)
        """
        self.domain_weight = domain_weight
        self.recency_weight = recency_weight
        self.content_weight = content_weight

        # Validate weights sum to 1.0
        total = domain_weight + recency_weight + content_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing")
            self.domain_weight /= total
            self.recency_weight /= total
            self.content_weight /= total

    def score(
        self,
        url: str,
        title: str = "",
        content: str = "",
        published_date: Optional[str] = None,
    ) -> Dict:
        """
        Calculate credibility score for a source.

        Components (weighted):
        - Domain credibility: Reputation of the source domain
        - Recency: How recent the content is
        - Content quality: Signals from title and content

        Args:
            url: Source URL
            title: Page title
            content: Page content (can be excerpt)
            published_date: Publication date (ISO format or parseable string)

        Returns:
            Dictionary with:
            - score: float (0-1) overall credibility
            - domain_score: float (0-1)
            - recency_score: float (0-1)
            - content_score: float (0-1)
            - tier: "high" | "medium" | "low"
            - signals: List[str] explanations
            - domain: str parsed domain
        """
        signals: List[str] = []

        # Parse domain
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
        except Exception:
            domain = ""
            path = ""
            signals.append("Invalid URL format")

        # 1. Domain credibility
        domain_score = self._score_domain(domain, path, signals)

        # 2. Recency
        recency_score = self._score_recency(published_date, signals)

        # 3. Content quality
        content_score = self._score_content(title, content, url, signals)

        # Apply spam penalty
        spam_penalty = self._check_spam_signals(url, title, signals)

        # Weighted total
        total_score = (
            domain_score * self.domain_weight +
            recency_score * self.recency_weight +
            content_score * self.content_weight
        ) - spam_penalty

        # Clamp to valid range
        total_score = max(0.0, min(1.0, total_score))

        # Determine tier
        if total_score >= 0.7:
            tier = "high"
        elif total_score >= 0.4:
            tier = "medium"
        else:
            tier = "low"

        result = {
            "score": round(total_score, 2),
            "domain_score": round(domain_score, 2),
            "recency_score": round(recency_score, 2),
            "content_score": round(content_score, 2),
            "tier": tier,
            "signals": signals,
            "domain": domain,
        }

        logger.debug(f"Credibility score for {domain}: {total_score:.2f} ({tier})")

        return result

    def _score_domain(self, domain: str, path: str, signals: List[str]) -> float:
        """
        Score based on domain reputation.

        Args:
            domain: Parsed domain name
            path: URL path
            signals: List to append signal explanations

        Returns:
            Score from 0.0 to 1.0
        """
        if not domain:
            signals.append("No domain found")
            return 0.3

        # Check high credibility domains
        for high_domain in HIGH_CREDIBILITY_DOMAINS:
            if high_domain.startswith("."):
                # TLD check
                if domain.endswith(high_domain):
                    signals.append(f"High-credibility TLD: {high_domain}")
                    return 1.0
            else:
                # Domain check
                if high_domain in domain:
                    signals.append(f"High-credibility source: {high_domain}")
                    return 1.0

        # Check medium credibility domains
        for med_domain in MEDIUM_CREDIBILITY_DOMAINS:
            if med_domain in domain:
                signals.append(f"Reputable source: {med_domain}")
                return 0.7

        # Check low credibility domains
        for low_domain in LOW_CREDIBILITY_DOMAINS:
            if low_domain in domain:
                # Reddit/forums are useful for user sentiment
                if "reddit.com" in domain:
                    signals.append("Community discussion (Reddit)")
                elif "quora.com" in domain:
                    signals.append("Q&A platform (Quora)")
                elif any(social in domain for social in ["twitter", "x.com", "linkedin", "facebook"]):
                    signals.append("Social media source")
                else:
                    signals.append(f"User-generated content: {low_domain}")
                return 0.4

        # Check for company blog patterns
        if "blog." in domain or "/blog" in path:
            signals.append("Company/personal blog")
            return 0.5

        # Check for documentation patterns
        if "docs." in domain or "/docs" in path or "/documentation" in path:
            signals.append("Documentation site")
            return 0.7

        # Unknown domain - assume moderate credibility
        signals.append(f"Unknown domain: {domain}")
        return 0.5

    def _score_recency(self, published_date: Optional[str], signals: List[str]) -> float:
        """
        Score based on content recency.

        Args:
            published_date: Publication date string
            signals: List to append signal explanations

        Returns:
            Score from 0.0 to 1.0
        """
        if not published_date:
            signals.append("No publication date available")
            return 0.5

        try:
            # Parse the date (handles various formats)
            pub_date = date_parser.parse(published_date, fuzzy=True)

            # Make timezone-aware if needed
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_days = (now - pub_date).days

            # Future dates are suspicious
            if age_days < 0:
                signals.append("Future publication date (suspicious)")
                return 0.3

            # Score based on age
            if age_days <= 30:
                signals.append(f"Very recent (published {age_days} days ago)")
                return 1.0
            elif age_days <= 90:
                signals.append(f"Recent (published {age_days} days ago)")
                return 0.9
            elif age_days <= 180:
                signals.append(f"Published {age_days} days ago")
                return 0.8
            elif age_days <= 365:
                signals.append(f"Published ~{age_days // 30} months ago")
                return 0.7
            elif age_days <= 730:
                signals.append(f"Published ~{age_days // 365} year(s) ago")
                return 0.5
            elif age_days <= 1095:
                signals.append(f"Published ~{age_days // 365} years ago")
                return 0.4
            else:
                years = age_days // 365
                signals.append(f"Dated content ({years}+ years old)")
                return 0.3

        except (ValueError, TypeError) as e:
            signals.append(f"Could not parse publication date")
            logger.debug(f"Date parse error: {e}")
            return 0.5

    def _score_content(
        self,
        title: str,
        content: str,
        url: str,
        signals: List[str],
    ) -> float:
        """
        Score based on content quality signals.

        Args:
            title: Page title
            content: Page content
            url: Source URL
            signals: List to append signal explanations

        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.5

        # Handle None values
        title = title or ""
        content = content or ""

        combined = f"{title} {content}".lower()
        url_lower = url.lower()

        # === Positive signals ===

        # Research/analysis indicators
        research_terms = ["study", "research", "analysis", "survey", "report", "findings", "data shows"]
        if any(term in combined for term in research_terms):
            score += 0.15
            signals.append("Contains research/analysis")

        # Statistics and data
        if re.search(r'\d+%|\$[\d,]+|[\d,]+ (users|customers|companies)', combined):
            score += 0.1
            signals.append("Contains statistics/data")

        # Product/pricing information (useful for competitive research)
        product_terms = ["pricing", "features", "comparison", "vs", "versus", "alternative"]
        if any(term in combined for term in product_terms):
            score += 0.1
            signals.append("Contains product/pricing info")

        # Expert indicators
        expert_terms = ["expert", "specialist", "analyst", "professor", "phd", "ceo", "founder"]
        if any(term in combined for term in expert_terms):
            score += 0.1
            signals.append("Expert source/citation")

        # Substantial content
        word_count = len(content.split()) if content else 0
        if word_count > 1000:
            score += 0.1
            signals.append("Comprehensive content")
        elif word_count > 500:
            score += 0.05
            signals.append("Substantial content")

        # Citations/references
        if re.search(r'\[\d+\]|references|sources|bibliography|cited', combined):
            score += 0.1
            signals.append("Contains citations/references")

        # === Negative signals ===

        # Sponsored/promotional content
        promo_terms = ["sponsored", "advertisement", "partner content", "paid post", "affiliate"]
        if any(term in combined for term in promo_terms):
            score -= 0.2
            signals.append("Sponsored/promotional content")

        # Error pages
        error_terms = ["404", "not found", "page removed", "no longer available", "error"]
        if any(term in title.lower() for term in error_terms):
            score -= 0.4
            signals.append("Page error detected")

        # Thin content
        if word_count < 100 and word_count > 0:
            score -= 0.1
            signals.append("Thin content")

        # Clickbait indicators
        clickbait_terms = ["you won't believe", "shocking", "mind-blowing", "secret", "hack", "trick"]
        if any(term in combined for term in clickbait_terms):
            score -= 0.1
            signals.append("Clickbait language detected")

        # Listicle without substance (e.g., "10 Best...")
        if re.match(r'^\d+\s+(best|top|worst|amazing)', title.lower()):
            score -= 0.05
            signals.append("Listicle format")

        return max(0.0, min(1.0, score))

    def _check_spam_signals(self, url: str, title: str, signals: List[str]) -> float:
        """
        Check for spam indicators and return penalty.

        Args:
            url: Source URL
            title: Page title
            signals: List to append signal explanations

        Returns:
            Penalty to subtract from score (0.0 to 0.3)
        """
        penalty = 0.0
        url_lower = url.lower()

        # Check URL for spam patterns
        for indicator in SPAM_INDICATORS:
            if indicator in url_lower:
                penalty += 0.1
                signals.append(f"Spam indicator in URL: {indicator}")
                break  # Only penalize once for URL

        # Excessive hyphens in domain (common in spam)
        domain = urlparse(url).netloc
        if domain.count("-") > 3:
            penalty += 0.1
            signals.append("Suspicious domain format")

        # Very long URLs often indicate tracking/spam
        if len(url) > 300:
            penalty += 0.05
            signals.append("Unusually long URL")

        return min(0.3, penalty)  # Cap penalty

    def score_batch(self, sources: List[Dict]) -> List[Dict]:
        """
        Score multiple sources and return sorted by credibility.

        Args:
            sources: List of dicts with url, title, content, published_date

        Returns:
            List of sources with credibility scores, sorted by score descending
        """
        scored = []

        for source in sources:
            result = self.score(
                url=source.get("url", ""),
                title=source.get("title", ""),
                content=source.get("content", source.get("snippet", "")),
                published_date=source.get("published_date"),
            )

            # Merge original source data with score
            scored_source = {**source, "credibility": result}
            scored.append(scored_source)

        # Sort by score descending
        scored.sort(key=lambda x: x["credibility"]["score"], reverse=True)

        return scored


def score_credibility(
    url: str,
    title: str = "",
    content: str = "",
    published_date: Optional[str] = None,
) -> Dict:
    """
    Convenience function for credibility scoring.

    Args:
        url: Source URL
        title: Page title
        content: Page content
        published_date: Publication date string

    Returns:
        Credibility score dictionary
    """
    scorer = CredibilityScorer()
    return scorer.score(url, title, content, published_date)


def score_sources(sources: List[Dict]) -> List[Dict]:
    """
    Score and sort multiple sources by credibility.

    Args:
        sources: List of source dictionaries

    Returns:
        Sources with credibility scores, sorted by score descending
    """
    scorer = CredibilityScorer()
    return scorer.score_batch(sources)
