import argparse
from pathlib import Path

from .llm import LLM
from .loop import idea_loop
from .report import Report


def main():
    p = argparse.ArgumentParser(description="IDEA: autonomous research loop")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run a research loop")
    run.add_argument("seed", help="Research question or hypothesis")
    run.add_argument("-w", "--workspace", default="./idea_runs")
    run.add_argument("-m", "--model", default="claude-opus-4-6")
    run.add_argument("--max-cost", type=float, default=50.0)
    run.add_argument("--max-depth", type=int, default=2)
    run.add_argument("--max-branches", type=int, default=3)
    run.add_argument("--no-web-search", action="store_true")

    rpt = sub.add_parser("report", help="View a saved report")
    rpt.add_argument("path")

    args = p.parse_args()
    if args.cmd == "run":
        _run(args)
    elif args.cmd == "report":
        print(Report.load(Path(args.path)))
    else:
        p.print_help()


def _run(args):
    llm = LLM(model=args.model, max_cost=args.max_cost)
    ws = Path(args.workspace)
    print("IDEA Research Loop")
    print(f"  Seed: {args.seed}")
    print(f"  Model: {args.model}, Budget: ${args.max_cost}")
    print(f"  Depth: {args.max_depth}, Branches: {args.max_branches}")
    print(f"  Web search: {'yes' if not args.no_web_search else 'no'}")
    print()
    report = idea_loop(
        args.seed, ws, llm,
        max_depth=args.max_depth, max_branches=args.max_branches,
        web_search=not args.no_web_search,
    )
    print(report)
    print(f"\nReport saved to: {ws}/report.json")


if __name__ == "__main__":
    main()
