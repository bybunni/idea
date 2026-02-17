"""D — Deep research: Literature review, baseline identification, dataset selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..llm import LLM

log = logging.getLogger(__name__)

SYSTEM = """\
You are a research scientist conducting a thorough literature review.

Given a hypothesis, search for and analyze:
1. The closest prior work — what already exists
2. The standard baseline approach for this problem
3. Standard datasets or benchmarks used in this area
4. Key metrics and how they're typically computed
5. Known pitfalls and failure modes
6. A concrete specification for implementing the baseline

Be specific. Name papers, methods, datasets. If you're uncertain, say so.
Do NOT hallucinate citations — if you don't know a specific paper, describe the approach
without inventing a citation.

Respond ONLY with JSON:
{
    "prior_work_summary": "paragraph summarizing what exists",
    "baseline_approach": "specific description of the standard baseline method",
    "baseline_spec": {
        "method": "name of baseline method",
        "description": "how to implement it",
        "expected_performance": "ballpark numbers if known"
    },
    "datasets": [
        {"name": "dataset name", "description": "what it is", "access": "how to get it"}
    ],
    "metrics": [
        {"name": "metric", "description": "what it measures", "direction": "higher_better|lower_better"}
    ],
    "pitfalls": ["known issue 1", "known issue 2"],
    "key_references": ["description of relevant work (no fabricated citations)"]
}
"""

# Web search tool definition for the Anthropic API
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
}


@dataclass
class LitReview:
    prior_work_summary: str
    baseline_approach: str
    baseline_spec: dict
    datasets: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    key_references: list[str] = field(default_factory=list)

    def prompt_block(self) -> str:
        parts = [
            f"## Literature Review",
            f"Prior work: {self.prior_work_summary}",
            f"\nBaseline: {self.baseline_approach}",
            f"Baseline spec: {self.baseline_spec}",
            f"\nDatasets: {self.datasets}",
            f"Metrics: {self.metrics}",
            f"Known pitfalls: {self.pitfalls}",
        ]
        return "\n".join(parts)


def deep_research_phase(
    llm: LLM,
    hypothesis: str,
    context: str = "",
    use_web_search: bool = True,
) -> LitReview:
    """Conduct deep research on the hypothesis."""
    user_msg = f"Hypothesis to investigate:\n{hypothesis}"
    if context:
        user_msg += f"\n\nContext from prior investigations:\n{context}"

    log.info(f"DEEP RESEARCH phase: investigating hypothesis")

    if use_web_search:
        # use tool-use loop with web search
        try:
            text = llm.with_tools(
                system=SYSTEM,
                user=user_msg,
                tools=[WEB_SEARCH_TOOL],
                temperature=0.5,
            )
            result = _parse_response(text)
        except Exception as e:
            log.warning(f"Web search failed ({e}), falling back to knowledge-only")
            result = llm(SYSTEM, user_msg, parse_json=True, temperature=0.5)
    else:
        result = llm(SYSTEM, user_msg, parse_json=True, temperature=0.5)

    if isinstance(result, dict) and "_parse_error" not in result:
        return LitReview(
            prior_work_summary=result.get("prior_work_summary", ""),
            baseline_approach=result.get("baseline_approach", ""),
            baseline_spec=result.get("baseline_spec", {}),
            datasets=result.get("datasets", []),
            metrics=result.get("metrics", []),
            pitfalls=result.get("pitfalls", []),
            key_references=result.get("key_references", []),
        )
    log.warning("Failed to parse lit review")
    return LitReview(
        prior_work_summary="Could not complete literature review.",
        baseline_approach="",
        baseline_spec={},
    )


def _parse_response(text: str) -> dict:
    """Extract JSON from a tool-use response that may contain mixed text."""
    import json

    # try direct parse
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"_raw": text, "_parse_error": True}
