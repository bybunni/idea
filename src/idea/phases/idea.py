"""I — Idea: Refine a seed idea into a testable hypothesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm import LLM

log = logging.getLogger(__name__)

SYSTEM = """\
You are a research scientist refining a raw idea into a precise, testable hypothesis.

You will receive a seed idea (possibly vague) and optionally context from prior investigations.

Your job:
1. Sharpen the idea into a single, falsifiable hypothesis
2. State the expected outcome if the hypothesis is true
3. State what would constitute a negative result
4. Identify what must be measured to test it
5. Suggest likely dependencies (Python packages) needed for experiments

Respond ONLY with JSON:
{
    "hypothesis": "precise testable statement",
    "expected_positive": "what we see if hypothesis is correct",
    "expected_negative": "what we see if hypothesis is wrong",
    "key_metrics": ["metric1", "metric2"],
    "dependencies": ["torch", "numpy", ...],
    "scope_notes": "any caveats or scope limits for a tractable experiment"
}
"""


@dataclass
class Hypothesis:
    hypothesis: str
    expected_positive: str
    expected_negative: str
    key_metrics: list[str]
    dependencies: list[str]
    scope_notes: str

    def prompt_block(self) -> str:
        return (
            f"## Hypothesis\n{self.hypothesis}\n\n"
            f"Expected positive result: {self.expected_positive}\n"
            f"Expected negative result: {self.expected_negative}\n"
            f"Key metrics: {', '.join(self.key_metrics)}\n"
            f"Scope: {self.scope_notes}"
        )


def idea_phase(llm: LLM, seed: str, context: str = "") -> Hypothesis:
    """Refine a seed idea into a testable hypothesis."""
    user_msg = f"Seed idea: {seed}"
    if context:
        user_msg += f"\n\nContext from prior investigations:\n{context}"

    log.info(f"IDEA phase: refining '{seed[:80]}...'")
    result = llm(SYSTEM, user_msg, parse_json=True, temperature=0.7)

    if isinstance(result, dict) and "_parse_error" not in result:
        return Hypothesis(
            hypothesis=result.get("hypothesis", seed),
            expected_positive=result.get("expected_positive", ""),
            expected_negative=result.get("expected_negative", ""),
            key_metrics=result.get("key_metrics", []),
            dependencies=result.get("dependencies", []),
            scope_notes=result.get("scope_notes", ""),
        )
    # fallback: use the seed directly
    log.warning("Failed to parse idea refinement, using seed directly")
    return Hypothesis(
        hypothesis=seed,
        expected_positive="",
        expected_negative="",
        key_metrics=[],
        dependencies=[],
        scope_notes="",
    )
