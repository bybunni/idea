"""Core IDEA loop: Idea -> Deep research -> Experiment -> Analyze."""

from __future__ import annotations

import logging
from pathlib import Path

from . import Config
from .llm import LLM, BudgetExhausted
from .report import Report
from .hypothesize import hypothesize
from .research import research
from .experiment import experiment
from .analyze import analyze, synthesize

log = logging.getLogger(__name__)


def idea_loop(
    seed: str,
    workspace: Path,
    llm: LLM,
    config: Config,
    *,
    context: str = "",
    depth: int = 0,
) -> Report:
    log.info(f"{'  ' * depth}IDEA loop depth={depth}: {seed[:80]}")
    workspace.mkdir(parents=True, exist_ok=True)
    report = Report(question=seed, depth=depth)

    try:
        if llm.exhausted:
            report.verdict = "budget_exceeded"
            report.cost = llm.total_cost
            report.save(workspace)
            return report

        # Phase 1: Hypothesize
        hyp = hypothesize(llm, seed, context)
        report.hypothesis = hyp.hypothesis
        log.info(f"{'  ' * depth}  Hypothesis: {hyp.hypothesis[:100]}")

        if llm.exhausted:
            report.verdict = "budget_exceeded"
            report.cost = llm.total_cost
            report.save(workspace)
            return report

        # Phase 2: Research
        lit = research(llm, hyp.prompt_block(), context, web_search=config.web_search)
        report.lit_review = lit.get("prior_work", "")
        log.info(f"{'  ' * depth}  Research complete")

        if llm.exhausted:
            report.verdict = "budget_exceeded"
            report.cost = llm.total_cost
            report.save(workspace)
            return report

        # Phase 3: Experiment
        exp = experiment(llm, hyp.prompt_block(), lit, workspace)
        report.experiment = exp
        log.info(f"{'  ' * depth}  Experiment complete")

        if llm.exhausted:
            report.verdict = "budget_exceeded"
            report.cost = llm.total_cost
            report.save(workspace)
            return report

        # Phase 4: Analyze
        analysis = analyze(llm, hyp.prompt_block(), exp, context)
        report.analysis = analysis.get("analysis", "")
        report.verdict = analysis.get("verdict", "inconclusive")
        report.headline = analysis.get("headline", "")
        report.next_steps = analysis.get("next_steps", [])
        log.info(f"{'  ' * depth}  Verdict: {report.verdict}")

        # Recurse into child loops
        if depth < config.max_depth and report.next_steps and not llm.exhausted:
            branches = report.next_steps[: config.max_branches]
            child_context = context + "\n\n" + report.context_summary()

            for i, step in enumerate(branches):
                if llm.exhausted:
                    break
                child_ws = workspace / f"child_{i}"
                child = idea_loop(
                    step, child_ws, llm, config,
                    context=child_context, depth=depth + 1,
                )
                report.children.append(child)

            if report.children:
                child_summaries = [
                    {
                        "question": c.question,
                        "verdict": c.verdict,
                        "analysis": c.analysis,
                    }
                    for c in report.children
                ]
                updated = synthesize(llm, analysis, child_summaries)
                report.analysis = updated.get("analysis", report.analysis)
                report.verdict = updated.get("verdict", report.verdict)
                report.headline = updated.get("headline", report.headline)

    except BudgetExhausted:
        log.info(f"{'  ' * depth}  Budget exhausted")
        report.verdict = "budget_exceeded"
    except Exception as e:
        log.error(f"{'  ' * depth}  Loop failed: {e}", exc_info=True)
        report.verdict = "error"
        report.analysis = f"Loop failed: {e}"

    report.cost = llm.total_cost
    report.save(workspace)
    return report
