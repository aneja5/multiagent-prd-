"""
ResearcherAgent - executes research plan and collects evidence.

This agent takes queries from PlannerAgent, executes them using web search,
fetches full content, scores credibility, deduplicates, and stores evidence.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TaskID

from agents.base_agent import BaseAgent
from app.logger import get_logger
from app.state import Evidence, Query, State, Task
from tools.credibility import score_credibility
from tools.dedupe import deduplicate_evidence
from tools.fetch_url import ContentFetcher
from tools.web_search import WebSearchTool, WebSearchError

logger = get_logger(__name__)


class ResearcherAgent(BaseAgent):
    """
    Execute research queries and collect evidence.

    Process:
    1. Get pending queries from state.research_plan
    2. For each query:
       a. Search web (get 10 results)
       b. Filter by relevance
       c. Fetch top 3-5 full pages
       d. Score credibility
       e. Add to evidence list
    3. Deduplicate all evidence
    4. Update state
    5. Mark research task as done

    Attributes:
        max_results_per_query: Number of search results per query
        max_fetch_per_query: Number of URLs to fetch full content from
        parallel_fetches: Number of parallel URL fetches
        console: Rich console for output
    """

    # Configuration
    MAX_RESULTS_PER_QUERY = 10
    MAX_FETCH_PER_QUERY = 5
    PARALLEL_FETCHES = 3
    QUERY_DELAY_SECONDS = 0.5  # Rate limiting between queries
    MAX_CONTENT_LENGTH = 5000  # Max chars for full_text

    def __init__(self, name: str, client: OpenAI, console: Optional[Console] = None) -> None:
        """
        Initialize the researcher agent.

        Args:
            name: Agent identifier (typically "research")
            client: Configured OpenAI client instance
            console: Optional Rich console for output
        """
        super().__init__(name, client)
        self.console = console or Console()
        self.max_results_per_query = self.MAX_RESULTS_PER_QUERY
        self.max_fetch_per_query = self.MAX_FETCH_PER_QUERY

        # Initialize tools (will be created on first use)
        self._search_tool: Optional[WebSearchTool] = None
        self._content_fetcher: Optional[ContentFetcher] = None

    @property
    def search_tool(self) -> WebSearchTool:
        """Lazy initialization of search tool."""
        if self._search_tool is None:
            try:
                self._search_tool = WebSearchTool()
            except ValueError as e:
                logger.error(f"Failed to initialize WebSearchTool: {e}")
                raise
        return self._search_tool

    @property
    def content_fetcher(self) -> ContentFetcher:
        """Lazy initialization of content fetcher."""
        if self._content_fetcher is None:
            self._content_fetcher = ContentFetcher()
        return self._content_fetcher

    def run(self, state: State) -> State:
        """
        Execute the research plan.

        Args:
            state: Current shared state with research_plan populated

        Returns:
            Updated state with evidence collected

        Raises:
            ValueError: If no queries are available
        """
        self.logger.info("Starting research execution...")
        self._log_action(state, "started_research")

        # Get pending queries
        queries = state.get_pending_queries()

        if not queries:
            self.logger.info("No pending queries to execute")
            self._log_action(state, "skipped_research", {"reason": "no_pending_queries"})
            return state

        self.console.print(
            f"\n[bold cyan]Executing {len(queries)} research queries...[/bold cyan]\n"
        )

        # Execute research with progress tracking
        all_evidence = self._execute_all_queries(queries, state)

        # Deduplicate all evidence
        self.console.print("\n[yellow]Deduplicating evidence...[/yellow]")
        unique_evidence = self._deduplicate_evidence(all_evidence)

        self.logger.info(
            f"Collected {len(all_evidence)} items, {len(unique_evidence)} unique"
        )
        self.console.print(
            f"[green]✓ Collected {len(unique_evidence)} unique sources[/green]\n"
        )

        # Add evidence to state
        self._add_evidence_to_state(state, unique_evidence)

        # Mark research task as done
        self._update_task_status(state)

        self._log_action(
            state,
            "completed_research",
            {
                "total_collected": len(all_evidence),
                "unique_evidence": len(unique_evidence),
                "queries_executed": len(queries),
            }
        )

        return state

    def _execute_all_queries(
        self,
        queries: List[Query],
        state: State
    ) -> List[Dict[str, Any]]:
        """
        Execute all queries with progress tracking.

        Args:
            queries: List of pending queries
            state: Current state

        Returns:
            List of all collected evidence items
        """
        all_evidence: List[Dict[str, Any]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task_id = progress.add_task(
                "[cyan]Researching...",
                total=len(queries)
            )

            for query_obj in queries:
                # Update progress description
                query_preview = query_obj.text[:40] + "..." if len(query_obj.text) > 40 else query_obj.text
                progress.update(
                    task_id,
                    description=f"[cyan]{query_preview}"
                )

                # Execute single query
                try:
                    evidence = self._execute_query(query_obj, state)
                    all_evidence.extend(evidence)

                    # Mark query as done
                    state.mark_query_done(query_obj.id)

                except Exception as e:
                    self.logger.error(f"Failed to execute query '{query_obj.text}': {e}")
                    self._log_action(
                        state,
                        "query_failed",
                        {"query_id": query_obj.id, "error": str(e)}
                    )
                    # Continue with other queries

                progress.advance(task_id)

                # Rate limiting between queries
                time.sleep(self.QUERY_DELAY_SECONDS)

        return all_evidence

    def _execute_query(
        self,
        query_obj: Query,
        state: State
    ) -> List[Dict[str, Any]]:
        """
        Execute a single query and return evidence.

        Args:
            query_obj: Query to execute
            state: Current state (for context)

        Returns:
            List of evidence items from this query
        """
        query_text = query_obj.text
        query_id = query_obj.id
        category = query_obj.category

        self.logger.info(f"Executing: {query_text}")

        # Search
        try:
            search_results = self.search_tool.search(
                query=query_text,
                max_results=self.max_results_per_query,
                search_depth="advanced"
            )
        except WebSearchError as e:
            self.logger.warning(f"Search failed for '{query_text}': {e}")
            return []

        if not search_results:
            self.logger.warning(f"No results for: {query_text}")
            return []

        # Sort by relevance score
        search_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Fetch top N URLs and build evidence
        evidence = self._fetch_and_process_results(
            search_results[:self.max_fetch_per_query],
            query_id=query_id,
            category=category
        )

        self.logger.info(f"  → Collected {len(evidence)} items for query")

        return evidence

    def _fetch_and_process_results(
        self,
        search_results: List[Dict[str, Any]],
        query_id: str,
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch full content and process search results into evidence.

        Uses parallel fetching for efficiency.

        Args:
            search_results: Search results to process
            query_id: ID of the query these results came from
            category: Query category for type inference

        Returns:
            List of evidence items
        """
        evidence: List[Dict[str, Any]] = []

        # Use thread pool for parallel fetching
        with ThreadPoolExecutor(max_workers=self.PARALLEL_FETCHES) as executor:
            # Submit fetch tasks
            future_to_result = {
                executor.submit(
                    self._fetch_single_result,
                    result,
                    query_id,
                    category
                ): result
                for result in search_results
            }

            # Collect results as they complete
            for future in as_completed(future_to_result):
                original_result = future_to_result[future]
                try:
                    item = future.result()
                    if item is not None:
                        evidence.append(item)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to process {original_result.get('url', 'unknown')}: {e}"
                    )

        return evidence

    def _fetch_single_result(
        self,
        result: Dict[str, Any],
        query_id: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch and process a single search result.

        Args:
            result: Search result dict with url, title, snippet
            query_id: ID of the originating query
            category: Query category

        Returns:
            Evidence item dict or None if fetch failed
        """
        url = result.get("url", "")

        if not url:
            return None

        # Fetch full content
        content = self.content_fetcher.fetch(url)

        if not content or not content.get("success"):
            self.logger.debug(f"Failed to fetch: {url}")
            return None

        # Score credibility
        cred = score_credibility(
            url=url,
            title=result.get("title", ""),
            content=content.get("content", ""),
            published_date=result.get("published_date"),
        )

        # Determine evidence type from URL and category
        evidence_type = self._infer_type(category, url)

        # Build evidence item
        item = {
            "url": url,
            "title": result.get("title", content.get("title", "")),
            "snippet": result.get("snippet", content.get("excerpt", "")),
            "full_text": content.get("content", "")[:self.MAX_CONTENT_LENGTH],
            "type": evidence_type,
            "tags": [category],
            "credibility": cred,
            "query_id": query_id,
            "relevance_score": result.get("score", 0.5),
        }

        return item

    def _infer_type(self, category: str, url: str) -> str:
        """
        Infer evidence type from query category and URL.

        Args:
            category: Query category (competitor, pain_points, etc.)
            url: Source URL

        Returns:
            Evidence type string (article, forum, docs, pricing, review)
        """
        url_lower = url.lower()

        # Forum/community sites
        forum_domains = [
            "reddit.com", "stackexchange.com", "stackoverflow.com",
            "quora.com", "news.ycombinator.com", "community."
        ]
        if any(forum in url_lower for forum in forum_domains):
            return "forum"

        # Review sites
        review_domains = [
            "g2.com", "capterra.com", "trustpilot.com",
            "trustradius.com", "getapp.com", "softwareadvice.com"
        ]
        if any(review in url_lower for review in review_domains):
            return "review"

        # Documentation sites
        doc_indicators = [
            "docs.", "/docs/", "/documentation", "/help/",
            "/support/", "/guide/", "/tutorial/", "/api/"
        ]
        if any(doc in url_lower for doc in doc_indicators):
            return "docs"

        # Pricing pages
        pricing_indicators = ["/pricing", "/plans", "/cost", "/subscription"]
        if any(price in url_lower for price in pricing_indicators):
            return "pricing"

        # Default based on category
        category_type_map = {
            "competitor": "article",
            "pain_points": "forum",
            "workflow": "article",
            "compliance": "docs",
        }

        return category_type_map.get(category, "article")

    def _deduplicate_evidence(
        self,
        evidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate evidence using the dedupe tool.

        Args:
            evidence: List of evidence items

        Returns:
            Deduplicated list
        """
        if not evidence:
            return []

        try:
            # The dedupe tool expects items with url, title, snippet fields
            return deduplicate_evidence(evidence)
        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
            # Return original list if dedup fails
            return evidence

    def _add_evidence_to_state(
        self,
        state: State,
        evidence_items: List[Dict[str, Any]]
    ) -> None:
        """
        Add evidence items to state.

        Converts dict items to Evidence model objects.

        Args:
            state: State to update
            evidence_items: List of evidence dicts
        """
        for i, item in enumerate(evidence_items):
            evidence_id = f"E{len(state.evidence) + 1}"

            # Get credibility tier from the credibility dict
            cred_data = item.get("credibility", {})
            cred_tier = cred_data.get("tier", "medium")

            # Validate tier
            if cred_tier not in ("high", "medium", "low"):
                cred_tier = "medium"

            # Validate type
            evidence_type = item.get("type", "article")
            if evidence_type not in ("article", "forum", "docs", "pricing", "review"):
                evidence_type = "article"

            # Clamp relevance score
            relevance = item.get("relevance_score", 0.5)
            relevance = max(0.0, min(1.0, float(relevance)))

            # Create Evidence object
            try:
                evidence_obj = Evidence(
                    id=evidence_id,
                    url=item.get("url", ""),
                    title=item.get("title", "")[:500],  # Limit title length
                    type=evidence_type,
                    snippet=item.get("snippet", "")[:1000],  # Limit snippet
                    full_text=item.get("full_text", ""),
                    tags=item.get("tags", []),
                    credibility=cred_tier,
                    query_id=item.get("query_id", ""),
                    relevance_score=relevance,
                )
                state.evidence.append(evidence_obj)

            except Exception as e:
                self.logger.warning(
                    f"Failed to create Evidence object for {item.get('url', 'unknown')}: {e}"
                )

        self.logger.info(f"Added {len(evidence_items)} evidence items to state")

    def _update_task_status(self, state: State) -> None:
        """
        Update research task status on the task board.

        Args:
            state: State with task_board to update
        """
        for task in state.task_board:
            if task.owner == "research":
                task.status = "done"
                self.logger.debug(f"Marked task {task.id} as done")
                break

    def get_research_stats(self, state: State) -> Dict[str, Any]:
        """
        Get statistics about the research results.

        Args:
            state: State with evidence

        Returns:
            Dictionary with research statistics
        """
        if not state.evidence:
            return {
                "total_evidence": 0,
                "by_type": {},
                "by_credibility": {},
                "by_category": {},
            }

        # Count by type
        by_type: Dict[str, int] = {}
        for e in state.evidence:
            by_type[e.type] = by_type.get(e.type, 0) + 1

        # Count by credibility
        by_credibility: Dict[str, int] = {}
        for e in state.evidence:
            by_credibility[e.credibility] = by_credibility.get(e.credibility, 0) + 1

        # Count by category (from tags)
        by_category: Dict[str, int] = {}
        for e in state.evidence:
            for tag in e.tags:
                by_category[tag] = by_category.get(tag, 0) + 1

        return {
            "total_evidence": len(state.evidence),
            "by_type": by_type,
            "by_credibility": by_credibility,
            "by_category": by_category,
        }
