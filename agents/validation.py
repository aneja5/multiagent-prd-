"""ValidationAgent - validates PRD quality and citations.

This agent reviews the generated PRD to check:
1. Citation coverage (are claims backed by evidence?)
2. Weak claims (vague or unsupported statements)
3. Missing sections
4. Quality issues
"""

import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.base_agent import BaseAgent
from app.logger import get_logger
from app.state import State, Task

logger = get_logger(__name__)
console = Console()


class ValidationIssue(BaseModel):
    """Single validation issue found in the PRD."""

    section: str = Field(description="PRD section where issue was found")
    issue_type: str = Field(
        description="Type of issue: missing_citation, weak_claim, too_generic, missing_section, incomplete"
    )
    description: str = Field(description="Description of the issue")
    severity: str = Field(description="Severity: high, medium, or low")
    suggestion: str = Field(description="Suggestion for fixing the issue")


class ValidationReport(BaseModel):
    """Complete validation results."""

    issues: List[ValidationIssue] = Field(default_factory=list)
    citation_coverage: float = Field(ge=0.0, le=1.0, description="Citation coverage 0-1")
    quality_score: float = Field(ge=0.0, le=100.0, description="Quality score 0-100")
    recommendations: List[str] = Field(default_factory=list)
    passed: bool = Field(default=False, description="Whether validation passed")


class ValidationAgent(BaseAgent):
    """Validate PRD quality and citation coverage.

    Performs comprehensive validation:
    1. Citation coverage - are pain points/competitors cited?
    2. Claim strength - vague statements flagged
    3. Completeness - all sections present
    4. Specificity - generic language flagged
    5. Feature quality - MVP features linked to pain points
    6. Metrics quality - success metrics are measurable

    Attributes:
        name: Agent identifier ("validation")
        client: OpenAI client for API calls
    """

    # Required PRD sections
    REQUIRED_SECTIONS = [
        "product_name",
        "problem_statement",
        "target_users",
        "solution_overview",
        "value_proposition",
        "mvp_features",
        "success_metrics",
    ]

    # Sections that should have citations
    CITATION_SECTIONS = [
        "problem_statement",
        "pain_points",
        "competitors",
        "mvp_features",
        "solution_overview",
    ]

    # Generic terms to flag
    GENERIC_TERMS = [
        "better", "easier", "faster", "improve", "enhance", "streamline",
        "optimize", "leverage", "robust", "scalable", "innovative",
        "cutting-edge", "best-in-class", "world-class", "seamless",
    ]

    # Vague quantifiers to flag
    VAGUE_QUANTIFIERS = [
        "some", "many", "few", "several", "various", "numerous",
        "significant", "substantial", "considerable",
    ]

    # Quality thresholds
    QUALITY_THRESHOLD = 70.0  # Minimum quality score to pass
    CITATION_THRESHOLD = 0.5  # Minimum citation coverage to pass

    def __init__(self, name: str, client: OpenAI) -> None:
        """Initialize the ValidationAgent.

        Args:
            name: Agent identifier (typically "validation")
            client: Configured OpenAI client instance
        """
        super().__init__(name, client)
        self.logger = get_logger(__name__)

    def run(self, state: State) -> State:
        """Validate PRD quality and citations.

        Args:
            state: Current shared state containing PRD

        Returns:
            Updated state with quality_report populated

        Raises:
            Exception: If validation encounters a critical error
        """
        self.logger.info("Starting PRD validation")
        self._log_action(state, "started_validation")

        # Create task on task board
        task = Task(
            id=f"T-VALIDATION-{state.run_id[:8]}",
            owner="validation",
            status="doing",
            description="Validate PRD quality and citations"
        )
        state.task_board.append(task)

        try:
            console.print("\n[bold cyan]Validating PRD...[/bold cyan]\n")

            # Check if PRD exists
            if not state.prd.sections and not state.prd.notion_markdown:
                self.logger.warning("No PRD to validate")
                console.print("[yellow]No PRD found to validate[/yellow]")
                self._mark_task_done(state, task.id)
                return state

            issues: List[ValidationIssue] = []

            # 1. Check citation coverage
            citation_issues = self._check_citations(state)
            issues.extend(citation_issues)
            self.logger.debug(f"Found {len(citation_issues)} citation issues")

            # 2. Check for weak claims
            weak_claim_issues = self._check_weak_claims(state)
            issues.extend(weak_claim_issues)
            self.logger.debug(f"Found {len(weak_claim_issues)} weak claim issues")

            # 3. Check completeness
            completeness_issues = self._check_completeness(state)
            issues.extend(completeness_issues)
            self.logger.debug(f"Found {len(completeness_issues)} completeness issues")

            # 4. Check specificity
            specificity_issues = self._check_specificity(state)
            issues.extend(specificity_issues)
            self.logger.debug(f"Found {len(specificity_issues)} specificity issues")

            # 5. Check feature quality
            feature_issues = self._check_features(state)
            issues.extend(feature_issues)
            self.logger.debug(f"Found {len(feature_issues)} feature issues")

            # 6. Check metrics quality
            metrics_issues = self._check_metrics(state)
            issues.extend(metrics_issues)
            self.logger.debug(f"Found {len(metrics_issues)} metrics issues")

            # Calculate scores
            citation_coverage = self._calculate_citation_coverage(state)
            quality_score = self._calculate_quality_score(issues)
            passed = (
                quality_score >= self.QUALITY_THRESHOLD and
                citation_coverage >= self.CITATION_THRESHOLD
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(issues, citation_coverage, quality_score)

            # Create validation report
            report = ValidationReport(
                issues=issues,
                citation_coverage=citation_coverage,
                quality_score=quality_score,
                recommendations=recommendations,
                passed=passed
            )

            # Save to state
            state.quality_report = {
                "issues": [issue.model_dump() for issue in issues],
                "citation_coverage_pct": round(citation_coverage * 100, 1),
                "quality_score": round(quality_score, 1),
                "recommendations": recommendations,
                "passed": passed,
                "issue_counts": {
                    "high": len([i for i in issues if i.severity == "high"]),
                    "medium": len([i for i in issues if i.severity == "medium"]),
                    "low": len([i for i in issues if i.severity == "low"]),
                },
            }

            # Display results
            self._display_results(issues, citation_coverage, quality_score, passed, recommendations)

            # Update original validation task if exists
            for t in state.task_board:
                if t.owner == "validation" and t.id != task.id:
                    t.status = "done"

            self._mark_task_done(state, task.id)

            self._log_action(
                state,
                "completed_validation",
                details={
                    "quality_score": round(quality_score, 1),
                    "citation_coverage": round(citation_coverage * 100, 1),
                    "issues_found": len(issues),
                    "passed": passed
                }
            )

            self.logger.info(
                f"Validation complete: score={quality_score:.1f}, "
                f"citations={citation_coverage*100:.1f}%, issues={len(issues)}"
            )

            return state

        except Exception as e:
            self._mark_task_blocked(state, task.id)
            self.logger.error(f"Validation failed: {e}")
            self._log_action(state, f"validation_failed: {str(e)}")
            raise

    def _check_citations(self, state: State) -> List[ValidationIssue]:
        """Check citation coverage across PRD sections.

        Args:
            state: Current state

        Returns:
            List of citation-related issues
        """
        issues = []
        citation_map = state.prd.citation_map

        # Check each section that should have citations
        for section in self.CITATION_SECTIONS:
            citations = citation_map.get(section, [])

            if not citations:
                # Determine severity based on section importance
                if section in ["pain_points", "problem_statement"]:
                    severity = "high"
                elif section in ["competitors", "mvp_features"]:
                    severity = "medium"
                else:
                    severity = "low"

                issues.append(ValidationIssue(
                    section=section,
                    issue_type="missing_citation",
                    description=f"Section '{section}' has no evidence citations",
                    severity=severity,
                    suggestion=f"Add evidence links to support claims in {section}"
                ))

        # Check if pain points in insights have evidence
        for i, pp in enumerate(state.insights.pain_points):
            if hasattr(pp, 'evidence_ids'):
                evidence_ids = pp.evidence_ids
            elif isinstance(pp, dict):
                evidence_ids = pp.get("evidence_ids", [])
            else:
                evidence_ids = []

            if not evidence_ids:
                pp_name = pp.cluster_name if hasattr(pp, 'cluster_name') else pp.get('cluster_name', f'Pain point {i+1}')
                issues.append(ValidationIssue(
                    section="pain_points",
                    issue_type="missing_citation",
                    description=f"Pain point '{pp_name}' has no supporting evidence",
                    severity="medium",
                    suggestion="Link pain point to research evidence"
                ))

        # Check if competitors have evidence
        for i, comp in enumerate(state.insights.competitors):
            if hasattr(comp, 'evidence_ids'):
                evidence_ids = comp.evidence_ids
            elif isinstance(comp, dict):
                evidence_ids = comp.get("evidence_ids", [])
            else:
                evidence_ids = []

            if not evidence_ids:
                comp_name = comp.name if hasattr(comp, 'name') else comp.get('name', f'Competitor {i+1}')
                issues.append(ValidationIssue(
                    section="competitors",
                    issue_type="missing_citation",
                    description=f"Competitor '{comp_name}' has no supporting evidence",
                    severity="low",
                    suggestion="Link competitor analysis to research sources"
                ))

        return issues

    def _check_weak_claims(self, state: State) -> List[ValidationIssue]:
        """Check for weak or vague claims in the PRD.

        Args:
            state: Current state

        Returns:
            List of weak claim issues
        """
        issues = []
        sections = state.prd.sections

        # Check problem statement length and specificity
        problem = sections.get("problem_statement", "")
        if problem and len(problem) < 100:
            issues.append(ValidationIssue(
                section="problem_statement",
                issue_type="weak_claim",
                description="Problem statement is too brief (< 100 characters)",
                severity="medium",
                suggestion="Expand problem statement with specific details, metrics, and pain point references"
            ))

        # Check for vague quantifiers in problem statement
        if problem:
            for vague in self.VAGUE_QUANTIFIERS:
                if re.search(rf'\b{vague}\b', problem.lower()):
                    issues.append(ValidationIssue(
                        section="problem_statement",
                        issue_type="weak_claim",
                        description=f"Problem statement uses vague quantifier: '{vague}'",
                        severity="low",
                        suggestion="Replace vague quantifiers with specific numbers or percentages"
                    ))
                    break  # Only report once

        # Check value proposition for generic terms
        value_prop = sections.get("value_proposition", "")
        if value_prop:
            generic_found = [term for term in self.GENERIC_TERMS if term in value_prop.lower()]
            if generic_found:
                issues.append(ValidationIssue(
                    section="value_proposition",
                    issue_type="too_generic",
                    description=f"Value proposition uses generic terms: {', '.join(generic_found[:3])}",
                    severity="low",
                    suggestion="Replace generic language with specific, quantifiable claims"
                ))

        # Check solution overview
        solution = sections.get("solution_overview", "")
        if solution and len(solution) < 150:
            issues.append(ValidationIssue(
                section="solution_overview",
                issue_type="weak_claim",
                description="Solution overview is too brief",
                severity="medium",
                suggestion="Expand solution overview with specific approach and methodology"
            ))

        return issues

    def _check_completeness(self, state: State) -> List[ValidationIssue]:
        """Check all required sections are present and populated.

        Args:
            state: Current state

        Returns:
            List of completeness issues
        """
        issues = []
        sections = state.prd.sections

        for section in self.REQUIRED_SECTIONS:
            content = sections.get(section)

            if not content:
                issues.append(ValidationIssue(
                    section=section,
                    issue_type="missing_section",
                    description=f"Required section '{section}' is missing or empty",
                    severity="high",
                    suggestion=f"Generate content for the {section} section"
                ))
            elif isinstance(content, str) and len(content.strip()) < 20:
                issues.append(ValidationIssue(
                    section=section,
                    issue_type="incomplete",
                    description=f"Section '{section}' has minimal content",
                    severity="medium",
                    suggestion=f"Expand the {section} section with more detail"
                ))
            elif isinstance(content, list) and len(content) == 0:
                issues.append(ValidationIssue(
                    section=section,
                    issue_type="missing_section",
                    description=f"Section '{section}' list is empty",
                    severity="high",
                    suggestion=f"Add items to the {section} section"
                ))

        # Check if PRD markdown was generated
        if not state.prd.notion_markdown:
            issues.append(ValidationIssue(
                section="prd_markdown",
                issue_type="missing_section",
                description="PRD markdown document was not generated",
                severity="high",
                suggestion="Regenerate the PRD using the template"
            ))

        return issues

    def _check_specificity(self, state: State) -> List[ValidationIssue]:
        """Check for specific vs generic content.

        Args:
            state: Current state

        Returns:
            List of specificity issues
        """
        issues = []
        sections = state.prd.sections

        # Check target users specificity
        target_users = sections.get("target_users", "")
        if target_users:
            # Flag if too short or uses "users" generically
            if len(target_users) < 50:
                issues.append(ValidationIssue(
                    section="target_users",
                    issue_type="too_generic",
                    description="Target users description is too brief",
                    severity="medium",
                    suggestion="Add specific demographics, roles, use cases, and context"
                ))

            # Check for generic "users" without specificity
            if re.search(r'\busers\b', target_users.lower()) and not any(
                term in target_users.lower() for term in
                ["freelance", "small business", "enterprise", "developer", "designer", "manager"]
            ):
                issues.append(ValidationIssue(
                    section="target_users",
                    issue_type="too_generic",
                    description="Target users uses generic 'users' without specificity",
                    severity="low",
                    suggestion="Specify user roles, industries, or personas"
                ))

        # Check JTBD specificity
        jtbd = sections.get("jtbd", "")
        if jtbd and "want to" not in jtbd.lower() and "so i can" not in jtbd.lower():
            issues.append(ValidationIssue(
                section="jtbd",
                issue_type="too_generic",
                description="JTBD doesn't follow the standard format",
                severity="low",
                suggestion="Use format: 'When I [situation], I want to [motivation], So I can [outcome]'"
            ))

        # Check differentiators
        differentiators = sections.get("differentiators", [])
        if isinstance(differentiators, list):
            for i, diff in enumerate(differentiators):
                if isinstance(diff, str):
                    generic_found = [term for term in self.GENERIC_TERMS if term in diff.lower()]
                    if generic_found:
                        issues.append(ValidationIssue(
                            section="differentiators",
                            issue_type="too_generic",
                            description=f"Differentiator {i+1} uses generic language",
                            severity="low",
                            suggestion="Make differentiators concrete and defensible"
                        ))
                        break  # Only report once

        return issues

    def _check_features(self, state: State) -> List[ValidationIssue]:
        """Check feature quality and coverage.

        Args:
            state: Current state

        Returns:
            List of feature-related issues
        """
        issues = []
        sections = state.prd.sections

        # Check MVP features count
        mvp_features = sections.get("mvp_features", [])
        if isinstance(mvp_features, list):
            if len(mvp_features) < 3:
                issues.append(ValidationIssue(
                    section="mvp_features",
                    issue_type="incomplete",
                    description=f"Only {len(mvp_features)} MVP features defined (minimum 3 recommended)",
                    severity="high",
                    suggestion="Add more P0 features based on identified pain points"
                ))
            elif len(mvp_features) > 10:
                issues.append(ValidationIssue(
                    section="mvp_features",
                    issue_type="weak_claim",
                    description=f"Too many MVP features ({len(mvp_features)}) - scope creep risk",
                    severity="medium",
                    suggestion="Reduce MVP scope to 5-8 essential features"
                ))

            # Check if features have descriptions
            for i, feature in enumerate(mvp_features):
                if isinstance(feature, str) and " - " not in feature and len(feature) < 30:
                    issues.append(ValidationIssue(
                        section="mvp_features",
                        issue_type="incomplete",
                        description=f"MVP feature {i+1} lacks description",
                        severity="low",
                        suggestion="Add brief description to each feature: 'Feature name - description'"
                    ))
                    break  # Only report once

        # Check phase 2 features
        phase2_features = sections.get("phase2_features", [])
        if isinstance(phase2_features, list) and len(phase2_features) == 0:
            issues.append(ValidationIssue(
                section="phase2_features",
                issue_type="incomplete",
                description="No phase 2 features defined",
                severity="low",
                suggestion="Add P1 features for post-MVP roadmap"
            ))

        # Check non-goals
        non_goals = sections.get("non_goals", [])
        if isinstance(non_goals, list) and len(non_goals) == 0:
            issues.append(ValidationIssue(
                section="non_goals",
                issue_type="incomplete",
                description="No explicit non-goals defined",
                severity="medium",
                suggestion="Define what the product will NOT do to set clear boundaries"
            ))

        return issues

    def _check_metrics(self, state: State) -> List[ValidationIssue]:
        """Check success metrics quality.

        Args:
            state: Current state

        Returns:
            List of metrics-related issues
        """
        issues = []
        sections = state.prd.sections

        success_metrics = sections.get("success_metrics", [])

        if not success_metrics:
            issues.append(ValidationIssue(
                section="success_metrics",
                issue_type="missing_section",
                description="No success metrics defined",
                severity="high",
                suggestion="Define 5-7 measurable success metrics"
            ))
            return issues

        if isinstance(success_metrics, list):
            if len(success_metrics) < 3:
                issues.append(ValidationIssue(
                    section="success_metrics",
                    issue_type="incomplete",
                    description=f"Only {len(success_metrics)} metrics defined (minimum 3 recommended)",
                    severity="medium",
                    suggestion="Add more success metrics covering different aspects"
                ))

            # Check for measurability
            has_numbers = False
            for metric in success_metrics:
                if isinstance(metric, str):
                    # Check if metric contains numbers or percentages
                    if re.search(r'\d+', metric) or '%' in metric:
                        has_numbers = True
                        break

            if not has_numbers:
                issues.append(ValidationIssue(
                    section="success_metrics",
                    issue_type="weak_claim",
                    description="Success metrics lack specific numerical targets",
                    severity="medium",
                    suggestion="Add specific numbers, percentages, or timeframes to metrics"
                ))

        return issues

    def _calculate_citation_coverage(self, state: State) -> float:
        """Calculate what percentage of key sections have citations.

        Args:
            state: Current state

        Returns:
            Citation coverage as a float 0-1
        """
        citation_map = state.prd.citation_map

        if not citation_map:
            return 0.0

        # Count sections with citations
        cited_sections = 0
        total_sections = len(self.CITATION_SECTIONS)

        for section in self.CITATION_SECTIONS:
            citations = citation_map.get(section, [])
            if citations:
                cited_sections += 1

        return cited_sections / total_sections if total_sections > 0 else 0.0

    def _calculate_quality_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score based on issues.

        Args:
            issues: List of validation issues

        Returns:
            Quality score 0-100
        """
        base_score = 100.0

        # Severity penalties
        severity_penalties = {
            "high": 15,
            "medium": 8,
            "low": 3
        }

        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 5)
            base_score -= penalty

        return max(0.0, min(100.0, base_score))

    def _generate_recommendations(
        self,
        issues: List[ValidationIssue],
        citation_coverage: float,
        quality_score: float
    ) -> List[str]:
        """Generate actionable recommendations based on issues.

        Args:
            issues: List of validation issues
            citation_coverage: Citation coverage score
            quality_score: Quality score

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Group issues by severity
        high_severity = [i for i in issues if i.severity == "high"]
        medium_severity = [i for i in issues if i.severity == "medium"]

        # High severity recommendation
        if high_severity:
            recommendations.append(
                f"CRITICAL: Address {len(high_severity)} high-severity issue(s) before finalizing PRD"
            )

        # Citation recommendation
        if citation_coverage < 0.5:
            recommendations.append(
                f"Improve citation coverage (currently {citation_coverage*100:.0f}%) - "
                "add evidence links to strengthen claims"
            )

        # Specificity recommendation
        generic_issues = [i for i in issues if i.issue_type == "too_generic"]
        if generic_issues:
            recommendations.append(
                "Replace generic language with specific, measurable statements"
            )

        # Completeness recommendation
        missing_issues = [i for i in issues if i.issue_type in ["missing_section", "incomplete"]]
        if missing_issues:
            sections = list(set(i.section for i in missing_issues))
            recommendations.append(
                f"Complete missing/incomplete sections: {', '.join(sections[:3])}"
            )

        # Medium severity recommendation
        if medium_severity and not high_severity:
            recommendations.append(
                f"Consider addressing {len(medium_severity)} medium-severity issue(s) to improve quality"
            )

        # Positive recommendation if passing
        if quality_score >= 80 and citation_coverage >= 0.6:
            recommendations.append(
                "PRD quality is good - ready for stakeholder review"
            )
        elif quality_score >= 70:
            recommendations.append(
                "PRD is acceptable but could benefit from minor improvements"
            )

        # Ensure at least one recommendation
        if not recommendations:
            recommendations.append("Review issues and apply suggested fixes")

        return recommendations

    def _display_results(
        self,
        issues: List[ValidationIssue],
        citation_coverage: float,
        quality_score: float,
        passed: bool,
        recommendations: List[str]
    ) -> None:
        """Display validation results with rich formatting.

        Args:
            issues: List of validation issues
            citation_coverage: Citation coverage score
            quality_score: Quality score
            passed: Whether validation passed
            recommendations: List of recommendations
        """
        # Determine status color
        if passed:
            status_color = "green"
            status_text = "PASSED"
        elif quality_score >= 60:
            status_color = "yellow"
            status_text = "NEEDS IMPROVEMENT"
        else:
            status_color = "red"
            status_text = "FAILED"

        # Issue counts
        high_count = len([i for i in issues if i.severity == "high"])
        medium_count = len([i for i in issues if i.severity == "medium"])
        low_count = len([i for i in issues if i.severity == "low"])

        # Summary panel
        summary_text = (
            f"[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]\n"
            f"[bold]Quality Score:[/bold] {quality_score:.0f}/100\n"
            f"[bold]Citation Coverage:[/bold] {citation_coverage*100:.0f}%\n"
            f"[bold]Issues:[/bold] {high_count} high, {medium_count} medium, {low_count} low"
        )

        console.print(Panel.fit(
            summary_text,
            title="Validation Summary",
            border_style=status_color
        ))
        console.print()

        # Issues table (if any)
        if issues:
            table = Table(title="Validation Issues", show_lines=True)
            table.add_column("Section", style="cyan", width=20)
            table.add_column("Type", style="yellow", width=18)
            table.add_column("Severity", width=10)
            table.add_column("Description", style="white", width=40)

            # Sort by severity
            sorted_issues = sorted(
                issues,
                key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.severity, 3)
            )

            for issue in sorted_issues[:15]:  # Show top 15
                sev_style = {
                    "high": "[bold red]HIGH[/bold red]",
                    "medium": "[yellow]MEDIUM[/yellow]",
                    "low": "[dim]LOW[/dim]"
                }.get(issue.severity, issue.severity)

                desc = issue.description
                if len(desc) > 40:
                    desc = desc[:37] + "..."

                table.add_row(
                    issue.section,
                    issue.issue_type,
                    sev_style,
                    desc
                )

            console.print(table)

            if len(issues) > 15:
                console.print(f"[dim]... and {len(issues) - 15} more issues[/dim]")
            console.print()
        else:
            console.print("[green]✓ No validation issues found![/green]\n")

        # Recommendations
        if recommendations:
            console.print("[bold]Recommendations:[/bold]")
            for i, rec in enumerate(recommendations, 1):
                if "CRITICAL" in rec:
                    console.print(f"  [red]{i}. {rec}[/red]")
                elif "good" in rec.lower() or "ready" in rec.lower():
                    console.print(f"  [green]{i}. {rec}[/green]")
                else:
                    console.print(f"  {i}. {rec}")
            console.print()

    def _mark_task_done(self, state: State, task_id: str) -> None:
        """Mark a task as done on the task board.

        Args:
            state: Current state
            task_id: ID of task to mark done
        """
        for t in state.task_board:
            if t.id == task_id:
                t.status = "done"
                break

    def _mark_task_blocked(self, state: State, task_id: str) -> None:
        """Mark a task as blocked on the task board.

        Args:
            state: Current state
            task_id: ID of task to mark blocked
        """
        for t in state.task_board:
            if t.id == task_id:
                t.status = "blocked"
                break
