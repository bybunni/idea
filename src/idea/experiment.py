import json
from pathlib import Path

from .sandbox import run_code

SYSTEM = """\
You are an ML engineer implementing an experiment.

Requirements:
- Single self-contained Python file
- Print a JSON dict as the LAST line of stdout with metric results
  Example: {"accuracy": 0.85, "f1": 0.82, "loss": 0.34}
- Use only standard ML libraries (torch, numpy, sklearn, scipy, etc)
- Handle errors — don't crash silently
- Keep it tractable: smaller models/datasets that run in <5 minutes
- Print progress to stderr, keep stdout clean for metrics JSON"""

CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Complete Python script"},
        "description": {"type": "string"},
    },
    "required": ["code", "description"],
}


def _generate_and_run(llm, prompt, workdir):
    r = llm.structured(SYSTEM, prompt, CODE_SCHEMA, temperature=0.4)
    result = run_code(r.get("code", ""), workdir)
    if not result.success:  # one retry
        r = llm.structured(SYSTEM,
            f"Fix this code.\n\nCode:\n```python\n{r.get('code', '')}\n```\n\n"
            f"Error:\n{result.stderr[-3000:]}",
            CODE_SCHEMA, temperature=0.3)
        result = run_code(r.get("code", ""), workdir)
    return r.get("description", ""), result


def experiment(llm, hypothesis_block, lit_review, workdir):
    baseline_desc, baseline_result = _generate_and_run(llm,
        f"Implement the BASELINE (standard approach) for:\n{hypothesis_block}\n\n"
        f"Research context:\n{json.dumps(lit_review, indent=2)}",
        workdir / "baseline")
    print(f"  baseline: {'ok' if baseline_result.success else 'FAIL'} {baseline_result.metrics}")

    idea_desc, idea_result = _generate_and_run(llm,
        f"Implement the NOVEL IDEA (the hypothesis) for:\n{hypothesis_block}\n\n"
        f"Research context:\n{json.dumps(lit_review, indent=2)}\n\n"
        f"Baseline results: {json.dumps(baseline_result.metrics)}",
        workdir / "idea")
    print(f"  idea: {'ok' if idea_result.success else 'FAIL'} {idea_result.metrics}")

    return {
        "baseline_description": baseline_desc,
        "baseline_metrics": baseline_result.metrics,
        "baseline_success": baseline_result.success,
        "idea_description": idea_desc,
        "idea_metrics": idea_result.metrics,
        "idea_success": idea_result.success,
    }
