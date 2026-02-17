"""A — Analyze: Synthesize results into report + next steps."""

from __future__ import annotations

import json
import logging

from ..llm import LLM
from .idea import Hypothesis
from .deep_research import LitReview
from .experiment import ExperimentResult

log = logging.getLogger(__name__)

SYSTEM = """\
You are a senior research scientist writing the analysis for an automated experiment.

You will receive the full context of an IDEA loop:
- The original hypothesis
- Literature review
- Baseline implementation and results
- Idea implementation and results
- Iteration history (if any)

Your job:
1. Determine the VERDICT: did the hypothesis hold?
   - "positive": idea clearly outperformed baseline on key metrics
   - "negative": idea clearly underperformed or showed no advantage
   - "inconclusive": results are mixed, noisy, or the experiment was flawed
   - "budget_exceeded": couldn't complete the investigation

2. Write a concise ANALYSIS (2-4 paragraphs):
   - What was tested and how
   - What the results show
   - Why the results make sense (or don't)
   - Methodological caveats

3. Generate 0-3 NEXT STEPS: specific follow-up questions worth investigating.
   Each should be a concrete, self-contained research question that a child IDEA loop
   could tackle autonomously. Generate 0 if the result is definitive.
   BAD next step: "investigate further"
   GOOD next step: "Test whether holomorphic activations with bounded real part (tanh(Re) * e^(i*Im)) avoid the gradient explosion seen in unbounded e^(iz)"

Respond ONLY with JSON:
{
    "verdict": "positive|negative|inconclusive",
    "analysis": "the full analysis text",
    "headline": "one-sentence summary of the finding",
    "next_steps": ["specific question 1", "specific question 2"],
    "confidence": 0.0-1.0,
    "methodological_concerns": ["concern 1", "concern 2"]
}
"""


def analyze_phase(
    llm: LLM,
    hypothesis: Hypothesis,
    lit_review: LitReview,
    experiment: ExperimentResult,
    context: str = "",
) -> dict:
    """Synthesize all results into a final analysis."""
    log.info("ANALYZE phase: synthesizing results")

    iteration_log = ""
    for it in experiment.iterations:
        iteration_log += (
            f"\n### Iteration {it.iteration}\n"
            f"Result: {it.result.summary()}\n"
            f"Critic: {it.critic_diagnosis}\n"
        )

    user_msg = (
        f"## Hypothesis\n{hypothesis.prompt_block()}\n\n"
        f"## Literature Review\n{lit_review.prompt_block()}\n\n"
        f"## Baseline\n"
        f"Description: {experiment.baseline_description}\n"
        f"Success: {experiment.baseline_result.success}\n"
        f"Metrics: {json.dumps(experiment.baseline_result.metrics)}\n\n"
        f"## Idea Implementation\n"
        f"Description: {experiment.idea_description}\n"
        f"Success: {experiment.idea_result.success}\n"
        f"Metrics: {json.dumps(experiment.idea_result.metrics)}\n\n"
        f"## Iteration History\n{iteration_log}\n\n"
        f"## Experiment summary\n{json.dumps(experiment.summary(), indent=2)}"
    )
    if context:
        user_msg += f"\n\n## Context from parent investigation\n{context}"

    result = llm(SYSTEM, user_msg, parse_json=True, temperature=0.5)

    if isinstance(result, dict) and "_parse_error" not in result:
        return result

    log.warning("Failed to parse analysis, returning raw")
    return {
        "verdict": "inconclusive",
        "analysis": result.get("_raw", "Analysis failed") if isinstance(result, dict) else str(result),
        "headline": "Analysis phase failed to produce structured output",
        "next_steps": [],
        "confidence": 0.0,
        "methodological_concerns": ["Analysis output was not parseable"],
    }


def synthesize_reports(
    llm: LLM,
    parent_analysis: dict,
    child_analyses: list[dict],
) -> dict:
    """Fold child investigation results back into the parent report."""
    log.info(f"Synthesizing {len(child_analyses)} child reports into parent")

    SYNTH_SYSTEM = """\
You are synthesizing the results of a research tree. You have a parent investigation
and several child investigations that explored follow-up questions.

Produce an updated analysis that incorporates the child findings.
Keep the same JSON format as the analysis phase.

Respond ONLY with JSON:
{
    "verdict": "positive|negative|inconclusive",
    "analysis": "updated analysis incorporating child findings",
    "headline": "updated one-sentence summary",
    "next_steps": [],
    "confidence": 0.0-1.0,
    "methodological_concerns": []
}
"""

    user_msg = (
        f"## Parent analysis\n{json.dumps(parent_analysis, indent=2)}\n\n"
        f"## Child investigations\n"
    )
    for i, child in enumerate(child_analyses):
        user_msg += f"\n### Child {i + 1}\n{json.dumps(child, indent=2)}\n"

    result = llm(SYNTH_SYSTEM, user_msg, parse_json=True, temperature=0.5)
    if isinstance(result, dict) and "_parse_error" not in result:
        return result
    return parent_analysis  # fallback: just return parent unchanged
