"""Phase 3: Implement baseline + idea, run both."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .llm import LLM
from .sandbox import run_code

log = logging.getLogger(__name__)

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
        "description": {"type": "string", "description": "What this implements"},
    },
    "required": ["code", "description"],
}


def experiment(
    llm: LLM, hypothesis_block: str, lit_review: dict, workdir: Path
) -> dict:
    """Run baseline + idea experiments, return results dict."""
    # --- Baseline ---
    baseline_dir = workdir / "baseline"
    baseline = llm.structured(
        SYSTEM,
        f"Implement the BASELINE (standard approach) for:\n{hypothesis_block}\n\n"
        f"Research context:\n{json.dumps(lit_review, indent=2)}",
        CODE_SCHEMA,
        temperature=0.4,
    )
    baseline_result = run_code(baseline.get("code", ""), baseline_dir)
    if not baseline_result.success:
        fix = llm.structured(
            SYSTEM,
            f"This baseline code failed. Fix it.\n\n"
            f"Code:\n```python\n{baseline.get('code', '')}\n```\n\n"
            f"Error:\n{baseline_result.stderr[-3000:]}",
            CODE_SCHEMA,
            temperature=0.3,
        )
        baseline_result = run_code(fix.get("code", ""), baseline_dir)
    log.info(f"Baseline: rc={baseline_result.returncode} metrics={baseline_result.metrics}")

    # --- Idea ---
    idea_dir = workdir / "idea"
    idea = llm.structured(
        SYSTEM,
        f"Implement the NOVEL IDEA (the hypothesis) for:\n{hypothesis_block}\n\n"
        f"Research context:\n{json.dumps(lit_review, indent=2)}\n\n"
        f"Baseline results: {json.dumps(baseline_result.metrics)}",
        CODE_SCHEMA,
        temperature=0.4,
    )
    idea_result = run_code(idea.get("code", ""), idea_dir)
    if not idea_result.success:
        fix = llm.structured(
            SYSTEM,
            f"This idea code failed. Fix it.\n\n"
            f"Code:\n```python\n{idea.get('code', '')}\n```\n\n"
            f"Error:\n{idea_result.stderr[-3000:]}",
            CODE_SCHEMA,
            temperature=0.3,
        )
        idea_result = run_code(fix.get("code", ""), idea_dir)
    log.info(f"Idea: rc={idea_result.returncode} metrics={idea_result.metrics}")

    return {
        "baseline_description": baseline.get("description", ""),
        "baseline_metrics": baseline_result.metrics,
        "baseline_success": baseline_result.success,
        "idea_description": idea.get("description", ""),
        "idea_metrics": idea_result.metrics,
        "idea_success": idea_result.success,
    }
