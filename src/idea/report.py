"""Report — the only interface between nested loops."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Report:
    question: str
    hypothesis: str = ""
    lit_review: str = ""
    baseline_summary: str = ""
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    idea_summary: str = ""
    idea_metrics: dict[str, Any] = field(default_factory=dict)
    experiment_log: list[dict] = field(default_factory=list)
    analysis: str = ""
    verdict: str = ""  # "positive" | "negative" | "inconclusive" | "budget_exceeded"
    next_steps: list[str] = field(default_factory=list)
    children: list[Report] = field(default_factory=list)
    budget_summary: dict = field(default_factory=dict)
    depth: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def save(self, path: Path) -> Path:
        out = path / "report.json"
        out.write_text(json.dumps(asdict(self), indent=2, default=str))
        # also write a human-readable markdown version
        md = path / "report.md"
        md.write_text(self.to_markdown())
        return out

    def to_markdown(self, indent: int = 0) -> str:
        pfx = "#" * (indent + 1)
        lines = [
            f"{pfx} IDEA Report: {self.question}",
            f"*{self.timestamp}* | verdict: **{self.verdict}**",
            "",
            f"{pfx}# Hypothesis",
            self.hypothesis,
            "",
            f"{pfx}# Literature Review",
            self.lit_review,
            "",
            f"{pfx}# Baseline",
            self.baseline_summary,
            f"```json\n{json.dumps(self.baseline_metrics, indent=2)}\n```"
            if self.baseline_metrics
            else "_no baseline_",
            "",
            f"{pfx}# Idea Implementation",
            self.idea_summary,
            f"```json\n{json.dumps(self.idea_metrics, indent=2)}\n```"
            if self.idea_metrics
            else "_no results_",
            "",
            f"{pfx}# Analysis",
            self.analysis,
            "",
        ]
        if self.next_steps:
            lines.append(f"{pfx}# Next Steps")
            for i, s in enumerate(self.next_steps, 1):
                lines.append(f"{i}. {s}")
            lines.append("")
        if self.children:
            lines.append(f"{pfx}# Sub-investigations")
            for child in self.children:
                lines.append(child.to_markdown(indent + 1))
        if self.budget_summary:
            lines.append(f"{pfx}# Budget")
            lines.append(
                f"API calls: {self.budget_summary.get('api_calls', '?')} | "
                f"Cost: ${self.budget_summary.get('cost_usd', '?')} | "
                f"Time: {self.budget_summary.get('elapsed_seconds', '?')}s"
            )
        return "\n".join(lines)

    @staticmethod
    def budget_exceeded(question: str, context: list[str], depth: int) -> Report:
        return Report(
            question=question,
            verdict="budget_exceeded",
            analysis="Budget exhausted before completing investigation.",
            depth=depth,
        )

    def context_summary(self) -> str:
        """Compact summary for injection into child loops as context."""
        parts = [f"## Prior finding (depth {self.depth}): {self.question}"]
        if self.hypothesis:
            parts.append(f"Hypothesis: {self.hypothesis}")
        if self.verdict:
            parts.append(f"Verdict: {self.verdict}")
        if self.baseline_metrics and self.idea_metrics:
            parts.append(f"Baseline metrics: {self.baseline_metrics}")
            parts.append(f"Idea metrics: {self.idea_metrics}")
        if self.analysis:
            # truncate to keep context manageable
            parts.append(f"Key finding: {self.analysis[:500]}")
        return "\n".join(parts)
