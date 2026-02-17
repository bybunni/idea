"""CLI for the IDEA research loop."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from . import Config
from .llm import LLM
from .loop import idea_loop
from .report import Report


@click.group()
def main():
    """IDEA: Idea -> Deep research -> Experiment -> Analyze."""
    pass


@main.command()
@click.argument("seed")
@click.option("--workspace", "-w", default="./idea_runs", help="Run artifacts directory")
@click.option("--model", "-m", default="claude-opus-4-6", help="Anthropic model")
@click.option("--max-cost", type=float, default=50.0, help="Max cost in USD")
@click.option("--max-depth", type=int, default=2, help="Max recursion depth")
@click.option("--max-branches", type=int, default=3, help="Max branches per depth")
@click.option("--no-web-search", is_flag=True, default=False, help="Disable web search")
@click.option("--verbose", "-v", is_flag=True, default=False)
def run(seed, workspace, model, max_cost, max_depth, max_branches, no_web_search, verbose):
    """Run an IDEA loop on a seed research question."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
    )

    config = Config(
        model=model,
        max_cost=max_cost,
        max_depth=max_depth,
        max_branches=max_branches,
        web_search=not no_web_search,
    )
    llm = LLM(model=config.model, max_cost=config.max_cost)
    ws = Path(workspace)

    print(f"IDEA Research Loop")
    print(f"  Seed: {seed}")
    print(f"  Model: {model}")
    print(f"  Budget: ${max_cost}")
    print(f"  Depth: {max_depth}, Branches: {max_branches}")
    print(f"  Web search: {'yes' if not no_web_search else 'no'}")
    print()

    report = idea_loop(seed, ws, llm, config)
    print(report)
    print(f"\nReport saved to: {ws}/report.json")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def report(path):
    """Display a previously generated report."""
    r = Report.load(Path(path))
    print(r)


if __name__ == "__main__":
    main()
