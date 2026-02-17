"""CLI for the IDEA research loop."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.tree import Tree

from .budget import Budget
from .loop import idea_loop
from .report import Report

console = Console()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )


@click.group()
def main():
    """IDEA: Idea → Deep research → Experiment → Analyze.

    Autonomous research loop. Hand it a seed idea, walk away, read the report.
    """
    pass


@main.command()
@click.argument("seed")
@click.option("--workspace", "-w", type=click.Path(), default="./idea_runs",
              help="Base directory for run artifacts")
@click.option("--model", "-m", default="claude-sonnet-4-5-20250514",
              help="Anthropic model to use")
@click.option("--max-depth", type=int, default=2,
              help="Maximum recursion depth for sub-investigations")
@click.option("--max-branches", type=int, default=3,
              help="Maximum parallel branches at each depth")
@click.option("--max-iters", type=int, default=5,
              help="Maximum experiment refinement iterations per loop")
@click.option("--max-api-calls", type=int, default=200,
              help="Hard limit on total API calls")
@click.option("--max-cost", type=float, default=50.0,
              help="Hard limit on total cost in USD")
@click.option("--max-time", type=float, default=3600.0,
              help="Hard limit on wall clock time in seconds")
@click.option("--parallel/--sequential", default=True,
              help="Run branches in parallel or sequentially")
@click.option("--no-web-search", is_flag=True, default=False,
              help="Disable web search in deep research phase")
@click.option("--context", "-c", type=click.Path(exists=True), default=None,
              help="Path to a file with additional context (prior reports, notes)")
@click.option("--verbose", "-v", is_flag=True, default=False)
def run(
    seed: str,
    workspace: str,
    model: str,
    max_depth: int,
    max_branches: int,
    max_iters: int,
    max_api_calls: int,
    max_cost: float,
    max_time: float,
    parallel: bool,
    no_web_search: bool,
    context: str | None,
    verbose: bool,
):
    """Run an IDEA loop on a seed research question.

    Example:

        idea run "holomorphic activations outperform ReLU on complex-valued SAR data"
    """
    setup_logging(verbose)

    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    budget = Budget(
        max_api_calls=max_api_calls,
        max_wall_seconds=max_time,
        max_depth=max_depth,
        max_branches=max_branches,
        max_experiment_iters=max_iters,
        max_cost_usd=max_cost,
    )

    ctx = ""
    if context:
        ctx = Path(context).read_text()

    console.print(Panel(
        f"[bold]Seed:[/bold] {seed}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Budget:[/bold] {max_api_calls} calls, ${max_cost}, {max_time}s\n"
        f"[bold]Depth:[/bold] {max_depth}, [bold]Branches:[/bold] {max_branches}\n"
        f"[bold]Mode:[/bold] {'parallel' if parallel else 'sequential'}\n"
        f"[bold]Web search:[/bold] {'yes' if not no_web_search else 'no'}",
        title="🔬 IDEA Research Loop",
        border_style="blue",
    ))

    report = idea_loop(
        seed=seed,
        workspace=ws,
        budget=budget,
        model=model,
        context=ctx,
        parallel=parallel,
        use_web_search=not no_web_search,
    )

    _print_report(report)
    console.print(f"\n[dim]Full report saved to: {ws}/report.json[/dim]")


@main.command()
@click.argument("path", type=click.Path(exists=True))
def report(path: str):
    """Display a previously generated report.

    Example:

        idea report ./idea_runs/20250216_123456_root/report.json
    """
    p = Path(path)
    if p.is_dir():
        p = p / "report.json"
    if not p.exists():
        console.print(f"[red]No report found at {p}[/red]")
        sys.exit(1)

    data = json.loads(p.read_text())
    # reconstruct enough to display
    console.print(Panel(
        p.with_suffix(".md").read_text() if p.with_suffix(".md").exists()
        else json.dumps(data, indent=2),
        title="📄 IDEA Report",
    ))


def _print_report(r: Report, depth: int = 0) -> None:
    """Pretty-print a report tree to the console."""
    verdict_color = {
        "positive": "green",
        "negative": "red",
        "inconclusive": "yellow",
        "budget_exceeded": "dim",
        "error": "red bold",
    }.get(r.verdict, "white")

    console.print()
    console.print(Panel(
        f"[bold]{r.question}[/bold]\n\n"
        f"[dim]Hypothesis:[/dim] {r.hypothesis}\n\n"
        f"[dim]Verdict:[/dim] [{verdict_color}]{r.verdict}[/{verdict_color}]\n\n"
        f"[dim]Baseline metrics:[/dim] {json.dumps(r.baseline_metrics, indent=2) if r.baseline_metrics else 'none'}\n"
        f"[dim]Idea metrics:[/dim] {json.dumps(r.idea_metrics, indent=2) if r.idea_metrics else 'none'}\n\n"
        f"{r.analysis}\n\n"
        f"[dim]Budget:[/dim] {r.budget_summary}",
        title=f"{'  ' * depth}📊 Depth {r.depth}",
        border_style=verdict_color,
    ))

    if r.next_steps:
        tree = Tree("[bold]Next steps[/bold]")
        for s in r.next_steps:
            tree.add(s)
        console.print(tree)

    for child in r.children:
        _print_report(child, depth + 1)


if __name__ == "__main__":
    main()
