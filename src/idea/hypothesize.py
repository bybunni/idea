SYSTEM = """\
You are a research scientist refining a raw idea into a precise, testable hypothesis.

Given a seed idea and optional context from prior investigations, produce:
1. A single, falsifiable hypothesis
2. Expected outcomes if true vs false
3. Key metrics to measure
4. Scope notes for a tractable experiment (use only stdlib + standard ML libs)"""

SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "expected_positive": {"type": "string"},
        "expected_negative": {"type": "string"},
        "key_metrics": {"type": "array", "items": {"type": "string"}},
        "scope_notes": {"type": "string"},
    },
    "required": ["hypothesis", "expected_positive", "expected_negative",
                  "key_metrics", "scope_notes"],
}


def format_hypothesis(h):
    return (
        f"## Hypothesis\n{h['hypothesis']}\n\n"
        f"Expected positive: {h['expected_positive']}\n"
        f"Expected negative: {h['expected_negative']}\n"
        f"Key metrics: {', '.join(h.get('key_metrics', []))}\n"
        f"Scope: {h.get('scope_notes', '')}"
    )


def hypothesize(llm, question, context=""):
    user = f"Seed idea: {question}"
    if context:
        user += f"\n\nContext from prior investigations:\n{context}"
    r = llm.structured(SYSTEM, user, SCHEMA)
    r.setdefault("hypothesis", question)
    return r
