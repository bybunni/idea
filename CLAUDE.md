# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IDEA (Idea -> Deep research -> Experiment -> Analyze) is an autonomous research loop that investigates scientific hypotheses without human intervention. It takes a seed research question and runs a four-phase cycle: refine hypothesis, literature review, implement & run experiments, analyze results. The analyze phase can spawn recursive child loops for follow-up questions.

## Commands

```bash
# Install (editable mode, registers `idea` CLI)
pip install -e .

# Run a research loop
idea run "your hypothesis here" --workspace ./runs

# View a previous report
idea report ./runs/
```

Requires `ANTHROPIC_API_KEY` environment variable. Default model: `claude-opus-4-6`.

No test suite, linter, or formatter is configured.

## Architecture

**Execution flow:** `cli.py` -> `loop.py:idea_loop()` -> four phases -> optional recursive child loops -> `Report`

The four phases execute sequentially within each loop iteration:
1. **`hypothesize.py`** — Refines seed text into a structured `Hypothesis` via `llm.structured()`
2. **`research.py`** — Literature review with optional web search; returns a dict
3. **`experiment.py`** — Generates baseline + novel Python scripts, executes them in `sandbox.py` subprocess, one retry on failure
4. **`analyze.py`** — Produces verdict (positive/negative/inconclusive) and 0-3 follow-up questions that spawn child loops

**Key design patterns:**
- **Structured output via forced `tool_use`** — `llm.structured()` uses `tool_choice={"type": "tool", "name": "respond"}` to force valid JSON. No regex parsing.
- **Report as universal interface** — Parent loops only see child `Report` objects. Reports serialize to JSON.
- **Single shared LLM instance** — One `LLM` object passed to all phases and child loops. Tracks `total_cost` globally. `exhausted` property checks budget.
- **Sequential execution** — No parallelism, no threads. Child loops run in a simple for-loop.
- **Config as simple dataclass** — `Config` in `__init__.py` holds model, max_cost, max_depth, max_branches, web_search.

**File layout:**
```
src/idea/
  __init__.py       # __version__ + Config dataclass
  cli.py            # Click CLI
  loop.py           # idea_loop() orchestration
  llm.py            # LLM wrapper: __call__, structured(), with_tools()
  sandbox.py        # run_code() + ExecResult
  report.py         # Report dataclass with save/load/__str__
  hypothesize.py    # Phase 1: seed -> Hypothesis
  research.py       # Phase 2: lit review + web search
  experiment.py     # Phase 3: baseline + idea, run both
  analyze.py        # Phase 4: verdict + follow-ups
```

**Infrastructure modules:**
- **`llm.py`** — Three methods: `__call__` (text), `structured` (forced tool_use JSON), `with_tools` (server-side tools like web search). Cost tracking via `total_cost` float. `BudgetExhausted` exception.
- **`sandbox.py`** — `run_code()` writes code to file, runs subprocess with 5-minute timeout, extracts JSON metrics from last stdout line.
- **`report.py`** — Dataclass with `save()` (JSON only), `load()`, `context_summary()`, and `__str__()` for readable text output.
