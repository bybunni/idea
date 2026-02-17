"""Phase 1: Refine seed idea into a testable hypothesis."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM

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
        "hypothesis": {"type": "string", "description": "Precise testable statement"},
        "expected_positive": {
            "type": "string",
            "description": "What we observe if hypothesis is correct",
        },
        "expected_negative": {
            "type": "string",
            "description": "What we observe if hypothesis is wrong",
        },
        "key_metrics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Metrics to measure",
        },
        "scope_notes": {
            "type": "string",
            "description": "Caveats and scope limits",
        },
    },
    "required": [
        "hypothesis",
        "expected_positive",
        "expected_negative",
        "key_metrics",
        "scope_notes",
    ],
}


@dataclass
class Hypothesis:
    hypothesis: str
    expected_positive: str
    expected_negative: str
    key_metrics: list[str]
    scope_notes: str

    def prompt_block(self) -> str:
        return (
            f"## Hypothesis\n{self.hypothesis}\n\n"
            f"Expected positive: {self.expected_positive}\n"
            f"Expected negative: {self.expected_negative}\n"
            f"Key metrics: {', '.join(self.key_metrics)}\n"
            f"Scope: {self.scope_notes}"
        )


def hypothesize(llm: LLM, question: str, context: str = "") -> Hypothesis:
    user = f"Seed idea: {question}"
    if context:
        user += f"\n\nContext from prior investigations:\n{context}"
    r = llm.structured(SYSTEM, user, SCHEMA)
    return Hypothesis(
        hypothesis=r.get("hypothesis", question),
        expected_positive=r.get("expected_positive", ""),
        expected_negative=r.get("expected_negative", ""),
        key_metrics=r.get("key_metrics", []),
        scope_notes=r.get("scope_notes", ""),
    )
