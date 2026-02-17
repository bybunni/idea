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
        "prior_work": {"type": "string"},
        "baseline_approach": {"type": "string"},
        "baseline_spec": {"type": "object"},
        "datasets": {"type": "array", "items": {"type": "object"}},
        "metrics": {"type": "array", "items": {"type": "object"}},
        "pitfalls": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prior_work", "baseline_approach", "baseline_spec",
                  "datasets", "metrics", "pitfalls"],
}

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


def research(llm, hypothesis, context="", web_search=True):
    user = f"Hypothesis to investigate:\n{hypothesis}"
    if context:
        user += f"\n\nContext from prior investigations:\n{context}"

    search_context = ""
    if web_search:
        try:
            search_context = llm.with_tools(
                SYSTEM, user, [WEB_SEARCH_TOOL], temperature=0.5)
        except Exception:
            pass

    if search_context:
        user += f"\n\nWeb search findings:\n{search_context}"

    return llm.structured(SYSTEM, user, SCHEMA, temperature=0.5)
