# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IDEA (Idea → Deep research → Experiment → Analyze) is an autonomous research loop that investigates scientific hypotheses without human intervention. It takes a seed research question and runs a four-phase cycle: refine hypothesis, literature review, implement & run experiments, analyze results. The analyze phase can spawn recursive child loops for follow-up questions.

## Commands

```bash
# Install (editable mode, registers `idea` CLI)
pip install -e .

# Run a research loop
idea run "your hypothesis here" --workspace ./runs

# View a previous report
idea report ./runs/
```

Requires `ANTHROPIC_API_KEY` environment variable. Default model: `claude-sonnet-4-5-20250514`.

No test suite, linter, or formatter is configured.

## Architecture

**Execution flow:** `cli.py` → `loop.py:idea_loop()` → four phases → optional recursive child loops → `Report`

The four phases execute sequentially within each loop iteration:
1. **`idea.py`** — Refines seed text into a structured `Hypothesis` (expected outcomes, metrics, dependencies)
2. **`deep_research.py`** — Literature review via multi-turn LLM tool-use with optional web search; produces `LitReview`
3. **`experiment.py`** — Generates baseline + novel Python scripts, executes them in `sandbox.py` subprocess, iterates with a separate critic LLM call (implementer/critic separation pattern)
4. **`analyze.py`** — Produces verdict (positive/negative/inconclusive) and 0-3 follow-up questions that spawn child loops

**Key design patterns:**
- **Report as universal interface** — Parent loops only see child `Report` objects, never child code. Reports serialize to JSON and markdown.
- **Budget as outer constraint** — `budget.py` tracks API calls, cost ($), wall time, and recursion depth with thread-safe counters. Child branches split the remaining budget equally.
- **Critic separation** — `experiment.py` uses separate implementer and critic LLM calls to avoid self-confirmation bias. The critic checks for bugs, overfitting, and fairness before results are accepted.
- **Recursive branching** — Follow-up questions from analyze phase spawn child `idea_loop()` calls. Runs in parallel (`ThreadPoolExecutor`) or sequential based on `--parallel` flag. Children's findings are synthesized back into the parent report.

**Infrastructure modules:**
- **`llm.py`** — Anthropic SDK wrapper with per-call cost tracking, multi-turn tool-use loop, and resilient JSON extraction (handles markdown fences, escaped content, partial JSON)
- **`sandbox.py`** — Subprocess execution with 5-minute timeout; extracts JSON metrics from last stdout line
- **`budget.py`** — Thread-safe budget with locks; supports `check()` to test exhaustion and `child(n)` to split budget
- **`report.py`** — Dataclass with `save()` (JSON + markdown), `load()`, and `context_summary()` for passing findings to children

## Known Issues

- `loop.py` imports `from .workspace import create_child_workspace` but the `workspace` module does not exist — this will cause a runtime `ImportError`
