"""Phase 2: Literature review with optional web search."""

from __future__ import annotations

from .llm import LLM

SYSTEM = """\
You are a research scientist conducting a literature review.

Given a hypothesis, analyze:
1. Closest prior work
2. Standard baseline approach
3. Datasets/benchmarks
4. Key metrics and how they're computed
5. Known pitfalls

Be specific. Don't hallucinate citations — if you don't know a specific paper,
describe the approach without inventing a citation."""

SCHEMA = {
    "type": "object",
    "properties": {
        "prior_work": {"type": "string", "description": "Summary of prior work"},
        "baseline_approach": {
            "type": "string",
            "description": "Standard baseline method",
        },
        "baseline_spec": {
            "type": "object",
            "description": "Specification for implementing the baseline",
        },
        "datasets": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Relevant datasets",
        },
        "metrics": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Key metrics",
        },
        "pitfalls": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Known pitfalls",
        },
    },
    "required": [
        "prior_work",
        "baseline_approach",
        "baseline_spec",
        "datasets",
        "metrics",
        "pitfalls",
    ],
}

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


def research(
    llm: LLM,
    hypothesis: str,
    context: str = "",
    web_search: bool = True,
) -> dict:
    user = f"Hypothesis to investigate:\n{hypothesis}"
    if context:
        user += f"\n\nContext from prior investigations:\n{context}"

    search_context = ""
    if web_search:
        try:
            search_context = llm.with_tools(
                SYSTEM, user, [WEB_SEARCH_TOOL], temperature=0.5
            )
        except Exception:
            pass

    if search_context:
        user += f"\n\nWeb search findings:\n{search_context}"

    return llm.structured(SYSTEM, user, SCHEMA, temperature=0.5)
