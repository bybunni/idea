import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Report:
    question: str
    hypothesis: str = ""
    lit_review: str = ""
    experiment: dict = field(default_factory=dict)
    analysis: str = ""
    verdict: str = ""
    headline: str = ""
    next_steps: list = field(default_factory=list)
    children: list = field(default_factory=list)
    cost: float = 0.0
    depth: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        out = path / "report.json"
        out.write_text(json.dumps(asdict(self), indent=2, default=str))
        return out

    @classmethod
    def load(cls, path):
        p = path / "report.json" if path.is_dir() else path
        return cls._from_dict(json.loads(p.read_text()))

    @classmethod
    def _from_dict(cls, d):
        children = [cls._from_dict(c) for c in d.pop("children", [])]
        return cls(**d, children=children)

    def context_summary(self):
        parts = [f"Prior finding (depth {self.depth}): {self.question}"]
        if self.hypothesis:
            parts.append(f"Hypothesis: {self.hypothesis}")
        if self.verdict:
            parts.append(f"Verdict: {self.verdict}")
        if self.headline:
            parts.append(f"Headline: {self.headline}")
        if self.experiment:
            parts.append(f"Experiment: {json.dumps(self.experiment)[:500]}")
        return "\n".join(parts)

    def __str__(self):
        lines = [
            "=" * 60,
            f"IDEA Report: {self.question}",
            f"Verdict: {self.verdict} | {self.headline}",
            f"Hypothesis: {self.hypothesis}",
            "",
            "Literature Review:",
            self.lit_review,
            "",
            "Experiment:",
            json.dumps(self.experiment, indent=2) if self.experiment else "(none)",
            "",
            "Analysis:",
            self.analysis,
        ]
        if self.next_steps:
            lines.append("\nNext Steps:")
            for i, s in enumerate(self.next_steps, 1):
                lines.append(f"  {i}. {s}")
        for child in self.children:
            lines.append(f"\n--- Child (depth {child.depth}) ---")
            lines.append(str(child))
        lines.append(f"\nCost: ${self.cost:.4f} | Depth: {self.depth} | {self.timestamp}")
        return "\n".join(lines)
