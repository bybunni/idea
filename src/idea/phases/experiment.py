"""E — Experiment: Implement baseline + idea, run, iterate."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import LLM
from ..budget import Budget
from ..sandbox import run_code, install_deps, ExecResult

log = logging.getLogger(__name__)

IMPLEMENT_SYSTEM = """\
You are an ML engineer implementing an experiment.

Requirements for ALL code you write:
- Single self-contained Python file
- All imports at the top
- Must print a JSON dict as the LAST line of stdout with metric results
  Example last line: {"accuracy": 0.85, "f1": 0.82, "loss": 0.34}
- Use only standard ML libraries (torch, numpy, sklearn, scipy, etc)
- Include error handling — don't crash silently
- If using a dataset that needs downloading, handle the download in the script
- Keep it tractable: prefer smaller models/datasets that run in <5 minutes
- Print progress to stderr so stdout stays clean for the final metrics JSON

You will be given:
- A hypothesis to test
- Literature review with baseline specification
- Whether to implement the BASELINE or the NOVEL IDEA

Respond ONLY with JSON:
{
    "code": "the complete Python script as a string",
    "description": "brief description of what this implements",
    "expected_runtime_seconds": 120,
    "dependencies": ["torch", "numpy"]
}
"""

REFINE_SYSTEM = """\
You are an ML engineer debugging and improving an experiment.

You have:
- The current code
- The execution results (stdout, stderr, metrics)
- A critic's diagnosis of what went wrong or could improve

Your job: produce an improved version of the code that addresses the diagnosis.
Keep the same structure (single file, JSON metrics on last line of stdout).

Respond ONLY with JSON:
{
    "code": "the complete improved Python script",
    "changes": "brief description of what you changed and why",
    "dependencies": ["torch", "numpy"]
}
"""

CRITIC_SYSTEM = """\
You are a skeptical experiment reviewer. Your job is to find problems.

You will receive:
- The hypothesis being tested
- Baseline code and its results
- Idea code and its results
- The iteration number

Evaluate honestly:
1. Did the code run correctly? Any bugs or measurement errors?
2. Is the comparison fair? Same data, same evaluation protocol?
3. Are the results statistically meaningful or just noise?
4. Is there overfitting, data leakage, or other methodological issues?
5. Has the idea genuinely outperformed the baseline?

Respond ONLY with JSON:
{
    "converged": true/false,
    "diagnosis": "what's happening — be specific",
    "issues": ["issue 1", "issue 2"],
    "suggested_changes": ["specific change 1", "specific change 2"],
    "confidence": 0.0-1.0,
    "verdict_if_final": "positive|negative|inconclusive"
}
"""


@dataclass
class ExperimentLog:
    iteration: int
    code_description: str
    result: ExecResult
    critic_diagnosis: str = ""


@dataclass
class ExperimentResult:
    baseline_code: str
    baseline_description: str
    baseline_result: ExecResult
    idea_code: str
    idea_description: str
    idea_result: ExecResult
    iterations: list[ExperimentLog] = field(default_factory=list)
    final_verdict: str = "inconclusive"

    def summary(self) -> dict:
        return {
            "baseline_metrics": self.baseline_result.metrics,
            "idea_metrics": self.idea_result.metrics,
            "iterations": len(self.iterations),
            "verdict": self.final_verdict,
            "baseline_success": self.baseline_result.success,
            "idea_success": self.idea_result.success,
        }


def experiment_phase(
    llm: LLM,
    hypothesis: str,
    lit_review_block: str,
    workspace: Path,
    budget: Budget,
) -> ExperimentResult:
    """Run the full experiment cycle: baseline → idea → iterate."""
    log.info("EXPERIMENT phase: starting")

    # --- Step 1: Implement and run baseline ---
    baseline_code, baseline_desc, baseline_deps = _implement(
        llm, hypothesis, lit_review_block, is_baseline=True
    )
    if baseline_deps:
        install_deps(baseline_deps, workspace)
    baseline_result = _run_with_retries(
        llm, baseline_code, workspace, "baseline.py", hypothesis, lit_review_block, budget
    )
    log.info(f"Baseline: {baseline_result.summary()}")

    # --- Step 2: Implement and run idea ---
    idea_context = (
        f"{lit_review_block}\n\n"
        f"## Baseline results\n"
        f"The baseline achieved: {json.dumps(baseline_result.metrics)}\n"
        f"Baseline code:\n```python\n{baseline_code}\n```"
    )
    idea_code, idea_desc, idea_deps = _implement(
        llm, hypothesis, idea_context, is_baseline=False
    )
    if idea_deps:
        install_deps(idea_deps, workspace)
    idea_result = _run_with_retries(
        llm, idea_code, workspace, "idea.py", hypothesis, idea_context, budget
    )
    log.info(f"Idea initial: {idea_result.summary()}")

    # --- Step 3: Iterate with critic ---
    iterations = []
    current_code = idea_code
    current_result = idea_result

    for i in range(budget.max_experiment_iters):
        if budget.exhausted():
            log.info("Budget exhausted during experiment iteration")
            break

        # Critic evaluates
        verdict = _critic(
            llm, hypothesis, baseline_code, baseline_result,
            current_code, current_result, i
        )
        iterations.append(ExperimentLog(
            iteration=i,
            code_description=idea_desc,
            result=current_result,
            critic_diagnosis=verdict.get("diagnosis", ""),
        ))

        if verdict.get("converged", False):
            log.info(f"Converged at iteration {i}: {verdict.get('diagnosis')}")
            return ExperimentResult(
                baseline_code=baseline_code,
                baseline_description=baseline_desc,
                baseline_result=baseline_result,
                idea_code=current_code,
                idea_description=idea_desc,
                idea_result=current_result,
                iterations=iterations,
                final_verdict=verdict.get("verdict_if_final", "positive"),
            )

        # Refine
        current_code, idea_desc, new_deps = _refine(
            llm, current_code, current_result, verdict
        )
        if new_deps:
            install_deps(new_deps, workspace)
        current_result = run_code(current_code, workspace, f"idea_v{i + 1}.py")
        log.info(f"Iteration {i}: {current_result.summary()}")

    # max iterations reached
    return ExperimentResult(
        baseline_code=baseline_code,
        baseline_description=baseline_desc,
        baseline_result=baseline_result,
        idea_code=current_code,
        idea_description=idea_desc,
        idea_result=current_result,
        iterations=iterations,
        final_verdict="inconclusive",
    )


def _implement(
    llm: LLM, hypothesis: str, context: str, is_baseline: bool
) -> tuple[str, str, list[str]]:
    """Have the LLM write experiment code."""
    role = "BASELINE (standard approach)" if is_baseline else "NOVEL IDEA (the hypothesis)"
    user_msg = (
        f"Implement the {role} for this hypothesis:\n{hypothesis}\n\n"
        f"Research context:\n{context}"
    )
    result = llm(IMPLEMENT_SYSTEM, user_msg, parse_json=True, temperature=0.4)
    if isinstance(result, dict) and "code" in result:
        return (
            result["code"],
            result.get("description", ""),
            result.get("dependencies", []),
        )
    # fallback: try to extract code from raw text
    raw = result.get("_raw", "") if isinstance(result, dict) else str(result)
    return raw, "implementation (parse failed)", []


def _run_with_retries(
    llm: LLM,
    code: str,
    workspace: Path,
    filename: str,
    hypothesis: str,
    context: str,
    budget: Budget,
    max_retries: int = 3,
) -> ExecResult:
    """Run code, and if it fails, have the LLM fix it."""
    result = run_code(code, workspace, filename)
    retries = 0
    while not result.success and retries < max_retries and not budget.exhausted():
        log.info(f"Code failed (attempt {retries + 1}), asking LLM to fix")
        fix_msg = (
            f"This code failed. Fix it.\n\n"
            f"Hypothesis: {hypothesis}\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f"Error:\n{result.stderr[-3000:]}\n\n"
            f"Context:\n{context}"
        )
        fix = llm(IMPLEMENT_SYSTEM, fix_msg, parse_json=True, temperature=0.3)
        if isinstance(fix, dict) and "code" in fix:
            code = fix["code"]
            deps = fix.get("dependencies", [])
            if deps:
                install_deps(deps, workspace)
        result = run_code(code, workspace, filename.replace(".py", f"_fix{retries}.py"))
        retries += 1
    return result


def _critic(
    llm: LLM,
    hypothesis: str,
    baseline_code: str,
    baseline_result: ExecResult,
    idea_code: str,
    idea_result: ExecResult,
    iteration: int,
) -> dict:
    """Separate critic evaluates the experiment."""
    user_msg = (
        f"## Hypothesis\n{hypothesis}\n\n"
        f"## Iteration: {iteration}\n\n"
        f"## Baseline code\n```python\n{baseline_code}\n```\n"
        f"## Baseline result\n{baseline_result.summary()}\n\n"
        f"## Idea code\n```python\n{idea_code}\n```\n"
        f"## Idea result\n{idea_result.summary()}"
    )
    result = llm(CRITIC_SYSTEM, user_msg, parse_json=True, temperature=0.3)
    if isinstance(result, dict) and "_parse_error" not in result:
        return result
    return {"converged": False, "diagnosis": "Critic failed to produce structured output"}


def _refine(
    llm: LLM, code: str, result: ExecResult, verdict: dict
) -> tuple[str, str, list[str]]:
    """Have the LLM improve the code based on critic feedback."""
    user_msg = (
        f"Current code:\n```python\n{code}\n```\n\n"
        f"Execution result:\n{result.summary()}\n\n"
        f"Critic diagnosis: {verdict.get('diagnosis', '')}\n"
        f"Issues: {verdict.get('issues', [])}\n"
        f"Suggested changes: {verdict.get('suggested_changes', [])}"
    )
    fix = llm(REFINE_SYSTEM, user_msg, parse_json=True, temperature=0.4)
    if isinstance(fix, dict) and "code" in fix:
        return fix["code"], fix.get("changes", ""), fix.get("dependencies", [])
    raw = fix.get("_raw", "") if isinstance(fix, dict) else str(fix)
    return raw, "refinement (parse failed)", []
