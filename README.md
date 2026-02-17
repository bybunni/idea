# IDEA

**I**dea → **D**eep research → **E**xperiment → **A**nalyze

Autonomous research loop. Hand it a seed idea and a budget. Walk away. Read the report.

## Install

```bash
pip install -e .
```

Requires `ANTHROPIC_API_KEY` in your environment.

## Usage

```bash
# Basic run
idea run "holomorphic activations outperform ReLU on complex-valued SAR data"

# With budget controls
idea run "Koopman spectral decomposition preserves transfer bounds" \
    --max-depth 2 \
    --max-branches 3 \
    --max-cost 25.0 \
    --max-time 7200 \
    --workspace ./runs/koopman

# Sequential mode (one branch at a time)
idea run "your idea" --sequential

# Inject prior context
idea run "follow-up question" --context ./prior_report.md

# Read a report
idea report ./idea_runs/
```

## How it works

```
seed idea
    │
    ▼
┌─────────┐
│ I: Idea  │  Refine seed → testable hypothesis
└────┬─────┘
     │
     ▼
┌───────────────┐
│ D: Deep Research │  Literature review, baselines, datasets
└────┬──────────┘
     │
     ▼
┌─────────────┐
│ E: Experiment │  Implement baseline + idea → run → critic → refine
│   ┌─────┐    │
│   │iterate│   │  (inner loop: run → critic → refine → run)
│   └─────┘    │
└────┬─────────┘
     │
     ▼
┌───────────┐
│ A: Analyze │  Verdict + next_steps
└────┬──────┘
     │
     ├── next_step_1 → IDEA loop (child)
     ├── next_step_2 → IDEA loop (child)
     └── next_step_3 → IDEA loop (child)
              │
              ▼
         synthesize children back into parent report
```

## Architecture

- **Budget** is the outer constraint. No human in the loop — the system runs until it converges or exhausts its budget (API calls, cost, wall time).
- **Critic separation**: The agent that runs the experiment is NOT the one evaluating it. A separate LLM call with a skeptical reviewer prompt checks for bugs, measurement errors, overfitting, and methodological issues.
- **Reports** are the only interface between nested loops. A parent never reads a child's code — it reads the child's report.
- **Parallel branches**: When the Analyze phase generates multiple next_steps, they fan out concurrently. Each child gets `1/n` of the remaining budget.

## Key files

```
idea/
├── cli.py          # click CLI
├── loop.py         # the core IDEA loop (THE thing)
├── budget.py       # hard limits on compute spend
├── report.py       # Report dataclass — the universal interface
├── llm.py          # thin Anthropic SDK wrapper with cost tracking
├── sandbox.py      # subprocess code execution with timeout
└── phases/
    ├── idea.py          # I: seed → hypothesis
    ├── deep_research.py # D: hypothesis → lit review + baseline spec
    ├── experiment.py    # E: implement + run + critic + iterate
    └── analyze.py       # A: synthesize → verdict + next_steps
```

## Budget defaults

| Parameter | Default | Flag |
|-----------|---------|------|
| API calls | 200 | `--max-api-calls` |
| Cost | $50 | `--max-cost` |
| Wall time | 1 hour | `--max-time` |
| Recursion depth | 2 | `--max-depth` |
| Branches per level | 3 | `--max-branches` |
| Experiment iterations | 5 | `--max-iters` |
