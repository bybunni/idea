from pathlib import Path

from .llm import LLM, BudgetExhausted
from .report import Report
from .hypothesize import hypothesize, format_hypothesis
from .research import research
from .experiment import experiment
from .analyze import analyze, synthesize


def idea_loop(seed, workspace, llm, *, context="", depth=0,
              max_depth=2, max_branches=3, web_search=True):
    indent = "  " * depth
    print(f"{indent}idea depth={depth}: {seed[:80]}")
    workspace.mkdir(parents=True, exist_ok=True)
    report = Report(question=seed, depth=depth)

    try:
        # Phase 1: Hypothesize
        hyp = hypothesize(llm, seed, context)
        report.hypothesis = hyp["hypothesis"]
        print(f"{indent}  hypothesis: {hyp['hypothesis'][:100]}")

        # Phase 2: Research
        hyp_block = format_hypothesis(hyp)
        lit = research(llm, hyp_block, context, web_search=web_search)
        report.lit_review = lit.get("prior_work", "")
        print(f"{indent}  research: done")

        # Phase 3: Experiment
        exp = experiment(llm, hyp_block, lit, workspace)
        report.experiment = exp
        print(f"{indent}  experiment: done")

        # Phase 4: Analyze
        analysis = analyze(llm, hyp_block, exp, context)
        report.analysis = analysis.get("analysis", "")
        report.verdict = analysis.get("verdict", "inconclusive")
        report.headline = analysis.get("headline", "")
        report.next_steps = analysis.get("next_steps", [])
        print(f"{indent}  verdict: {report.verdict}")

        # Recurse into child loops
        if depth < max_depth and report.next_steps and not llm.exhausted:
            branches = report.next_steps[:max_branches]
            child_context = context + "\n\n" + report.context_summary()
            for i, step in enumerate(branches):
                if llm.exhausted:
                    break
                child = idea_loop(
                    step, workspace / f"child_{i}", llm,
                    context=child_context, depth=depth + 1,
                    max_depth=max_depth, max_branches=max_branches,
                    web_search=web_search,
                )
                report.children.append(child)

            if report.children:
                child_summaries = [{"question": c.question, "verdict": c.verdict,
                                    "analysis": c.analysis} for c in report.children]
                updated = synthesize(llm, analysis, child_summaries)
                report.analysis = updated.get("analysis", report.analysis)
                report.verdict = updated.get("verdict", report.verdict)
                report.headline = updated.get("headline", report.headline)

    except BudgetExhausted:
        print(f"{indent}  budget exhausted")
        report.verdict = "budget_exceeded"
    except Exception as e:
        print(f"{indent}  error: {e}")
        report.verdict = "error"
        report.analysis = f"Loop failed: {e}"

    report.cost = llm.total_cost
    report.save(workspace)
    return report
