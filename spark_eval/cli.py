"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import format_report, get_spark, run_suite, write_report
from .schema import TaskValidationError, load_tasks

DEFAULT_TASKS = Path(__file__).resolve().parent.parent / "tasks"


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="task directory")
    p.add_argument("--category", action="append", dest="categories", help="filter (repeatable)")
    p.add_argument("--id", action="append", dest="ids", help="run specific task ids")


def cmd_run(args: argparse.Namespace) -> int:
    from .models import build_model  # local import: keeps `validate` dependency-light

    tasks = load_tasks(args.tasks, args.categories, args.ids)
    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 1

    model = build_model(args.model, timeout=args.request_timeout)
    ks = tuple(sorted({1, *(args.k or [])}))
    if args.n < max(ks):
        print(
            f"error: --n {args.n} is too small for pass@{max(ks)}; "
            f"the estimator needs n >= k",
            file=sys.stderr,
        )
        return 2

    report = run_suite(
        tasks,
        model,
        n_samples=args.n,
        ks=ks,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        keep_responses=not args.no_responses,
    )

    print(format_report(report))
    if args.out:
        write_report(report, args.out)
        print(f"wrote {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Structural checks only. `selfcheck` is the one that executes anything."""
    try:
        tasks = load_tasks(args.tasks, args.categories, args.ids)
    except TaskValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    from collections import Counter

    counts = Counter(t.category for t in tasks)
    print(f"{len(tasks)} tasks, all structurally valid\n")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<22} {n:>4}")
    missing = [t.id for t in tasks if not t.probes]
    if missing:
        print(f"\nwarning: {len(missing)} tasks have no 'probes' note: {missing[:5]}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Execute every reference solution.

    This is the check that matters. If a gold solution does not run, every
    model scored against that task gets a meaningless result.
    """
    from .harness import evaluate_candidate

    tasks = load_tasks(args.tasks, args.categories, args.ids)
    spark = get_spark("spark-eval-selfcheck")
    spark.sparkContext.setLogLevel("ERROR")

    failures = []
    for i, task in enumerate(tasks, 1):
        result = evaluate_candidate(spark, task, task.solution, timeout=args.timeout)
        status = "ok" if result.ok else f"BROKEN ({result.status})"
        print(f"[{i:>3}/{len(tasks)}] {task.id:<40} {status}", flush=True)
        if not result.ok:
            failures.append((task.id, result.detail))

    spark.stop()

    if failures:
        print(f"\n{len(failures)} reference solution(s) failed:\n", file=sys.stderr)
        for tid, detail in failures:
            print(f"  {tid}: {detail}", file=sys.stderr)
        return 1
    print(f"\nall {len(tasks)} reference solutions execute and self-compare")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spark-eval",
        description="Execution-based benchmark for PySpark code generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="score a model against the suite")
    _add_selection_args(p_run)
    p_run.add_argument(
        "--model",
        required=True,
        help="ollama:<tag> | openai:<model> | dummy:reference",
    )
    p_run.add_argument("--n", type=int, default=1, help="samples per task")
    p_run.add_argument(
        "--k", type=int, action="append", help="report pass@k (repeatable, needs n>=k)"
    )
    p_run.add_argument("--temperature", type=float, default=0.2)
    p_run.add_argument("--max-tokens", type=int, default=1024)
    p_run.add_argument("--timeout", type=int, default=60, help="per-task exec seconds")
    p_run.add_argument("--request-timeout", type=int, default=300)
    p_run.add_argument("--out", type=Path, help="write full JSON report here")
    p_run.add_argument(
        "--no-responses",
        action="store_true",
        help="omit raw generations from the report (smaller files)",
    )
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="structural check on task files")
    _add_selection_args(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_self = sub.add_parser("selfcheck", help="execute every reference solution")
    _add_selection_args(p_self)
    p_self.add_argument("--timeout", type=int, default=60)
    p_self.set_defaults(func=cmd_selfcheck)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
