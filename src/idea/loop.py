"""The core IDEA loop: Idea → Deep research → Experiment → Analyze."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .budget import Budget
from .llm import LLM, BudgetExhausted
from .report import Report
from .workspace import create_child_workspace
from .sandbox import install_deps

from .phases.idea import idea_phase
from .phases.deep_research import deep_research_phase
from .phases.experiment import experiment_phase
from .phases.analyze import analyze_phase, synthesize_reports

log = logging.getLogger(__name__)


def idea_loop(
    seed: str,
    workspace: Path,
    budget: Budget,
    *,
    model: str = "claude-sonnet-4-5-20250514",
    context: str = "",
    depth: int = 0,
    parallel: bool = True,
    use_web_search: bool = True,
) -> Report:
    """
    The atomic unit of research. Everything nests from this.

    I — Idea:          Refine seed into testable hypothesis
    D — Deep research: Literature review, find baselines + datasets
    E — Experiment:    Implement baseline + idea, run, iterate with critic
    A — Analyze:       Verdict, synthesis, generate next_steps

    If next_steps are generated and budget allows, recurse.
    """
    log.info(f"{'  ' * depth}IDEA loop depth={depth}: {seed[:80]}...")

    if budget.exhausted():
        log.info(f"{'  ' * depth}Budget exhausted at depth {depth}")
        return Report.budget_exceeded(seed, context, depth)

    report = Report(question=seed, depth=depth)

    try:
        llm = LLM(model=model, budget=budget)

        # ── I: Idea ──────────────────────────────────────────────
        hypothesis = idea_phase(llm, seed, context)
        report.hypothesis = hypothesis.hypothesis
        log.info(f"{'  ' * depth}  Hypothesis: {hypothesis.hypothesis[:100]}...")

        if budget.exhausted():
            report.verdict = "budget_exceeded"
            report.budget_summary = budget.summary()
            return report

        # ── D: Deep Research ─────────────────────────────────────
        lit = deep_research_phase(
            llm, hypothesis.prompt_block(), context, use_web_search=use_web_search
        )
        report.lit_review = lit.prior_work_summary
        log.info(f"{'  ' * depth}  Lit review complete, baseline: {lit.baseline_spec.get('method', '?')}")

        if budget.exhausted():
            report.verdict = "budget_exceeded"
            report.budget_summary = budget.summary()
            return report

        # install dependencies identified by I and D phases
        all_deps = list(set(hypothesis.dependencies))
        if all_deps:
            install_deps(all_deps, workspace)

        # ── E: Experiment ────────────────────────────────────────
        experiment = experiment_phase(
            llm, hypothesis.prompt_block(), lit.prompt_block(), workspace, budget
        )
        report.baseline_summary = experiment.baseline_description
        report.baseline_metrics = experiment.baseline_result.metrics or {}
        report.idea_summary = experiment.idea_description
        report.idea_metrics = experiment.idea_result.metrics or {}
        report.experiment_log = [
            {
                "iteration": it.iteration,
                "description": it.code_description,
                "success": it.result.success,
                "metrics": it.result.metrics,
                "diagnosis": it.critic_diagnosis,
            }
            for it in experiment.iterations
        ]
        log.info(f"{'  ' * depth}  Experiment complete: {experiment.summary()}")

        # ── A: Analyze ───────────────────────────────────────────
        analysis = analyze_phase(llm, hypothesis, lit, experiment, context)
        report.analysis = analysis.get("analysis", "")
        report.verdict = analysis.get("verdict", "inconclusive")
        report.next_steps = analysis.get("next_steps", [])
        log.info(
            f"{'  ' * depth}  Analysis: verdict={report.verdict}, "
            f"{len(report.next_steps)} next steps"
        )

        # ── Recurse ──────────────────────────────────────────────
        if (
            depth < budget.max_depth
            and report.next_steps
            and not budget.exhausted()
        ):
            branches = report.next_steps[: budget.max_branches]
            child_context = context + "\n\n" + report.context_summary()

            if parallel and len(branches) > 1:
                report.children = _run_parallel(
                    branches, workspace, budget, model, child_context,
                    depth, use_web_search
                )
            else:
                report.children = _run_sequential(
                    branches, workspace, budget, model, child_context,
                    depth, use_web_search
                )

            # Synthesize child findings back into parent
            if report.children:
                child_analyses = [
                    {
                        "question": c.question,
                        "verdict": c.verdict,
                        "analysis": c.analysis,
                        "metrics": c.idea_metrics,
                    }
                    for c in report.children
                ]
                updated = synthesize_reports(llm, analysis, child_analyses)
                report.analysis = updated.get("analysis", report.analysis)
                report.verdict = updated.get("verdict", report.verdict)

    except BudgetExhausted:
        log.info(f"{'  ' * depth}  Budget exhausted during loop")
        report.verdict = "budget_exceeded"
    except Exception as e:
        log.error(f"{'  ' * depth}  IDEA loop failed: {e}", exc_info=True)
        report.verdict = "error"
        report.analysis = f"Loop failed with error: {e}"

    report.budget_summary = budget.summary()
    report.save(workspace)
    return report


def _run_sequential(
    branches: list[str],
    workspace: Path,
    budget: Budget,
    model: str,
    context: str,
    depth: int,
    use_web_search: bool,
) -> list[Report]:
    """Run child branches one at a time."""
    children = []
    for i, step in enumerate(branches):
        if budget.exhausted():
            break
        child_ws = create_child_workspace(workspace, i)
        child_budget = budget.split(len(branches) - i)
        child = idea_loop(
            seed=step,
            workspace=child_ws,
            budget=child_budget,
            model=model,
            context=context,
            depth=depth + 1,
            parallel=False,  # children of sequential are also sequential
            use_web_search=use_web_search,
        )
        children.append(child)
    return children


def _run_parallel(
    branches: list[str],
    workspace: Path,
    budget: Budget,
    model: str,
    context: str,
    depth: int,
    use_web_search: bool,
) -> list[Report]:
    """Run child branches concurrently using threads.
    
    Note: We use ThreadPoolExecutor (not Process) because the LLM client
    and budget objects need to be shared. The actual parallelism comes from
    concurrent API calls, not CPU-bound work.
    """
    from concurrent.futures import ThreadPoolExecutor

    children = []
    child_budgets = [budget.split(len(branches)) for _ in branches]
    child_workspaces = [create_child_workspace(workspace, i) for i in range(len(branches))]

    def run_branch(args):
        step, ws, b = args
        return idea_loop(
            seed=step,
            workspace=ws,
            budget=b,
            model=model,
            context=context,
            depth=depth + 1,
            parallel=False,  # don't nest parallelism
            use_web_search=use_web_search,
        )

    with ThreadPoolExecutor(max_workers=min(len(branches), 4)) as executor:
        futures = {
            executor.submit(run_branch, (step, ws, b)): i
            for i, (step, ws, b) in enumerate(
                zip(branches, child_workspaces, child_budgets)
            )
        }
        for future in as_completed(futures):
            try:
                children.append(future.result())
            except Exception as e:
                log.error(f"Branch failed: {e}", exc_info=True)

    return children
