"""Execution harness: run candidate code against fixtures and judge the result.

This is the part that makes spark-eval different from string-similarity scoring.
Nothing here looks at the *text* of the generated code. It runs it and compares
the DataFrame that comes out.

Security note, stated plainly: `run_code` executes untrusted model output in
this process. The import guard below stops casual accidents (a model that
decides to `import os` and clean up after itself), not a determined adversary.
If you are scoring untrusted checkpoints, run the CLI inside a container with no
network and a read-only mount.
"""

from __future__ import annotations

import builtins
import math
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame, SparkSession

from .schema import Compare, Fixture, Task

# Modules a correct PySpark answer never needs. Blocking them turns "the model
# wandered off and deleted the fixtures" into a clean task failure.
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "httpx",
    "aiohttp",
    "pathlib",
    "ctypes",
    "importlib",
    "pickle",
    "multiprocessing",
    "tempfile",
    "glob",
}


class TaskTimeout(Exception):
    pass


class BlockedImport(Exception):
    pass


@dataclass
class ExecResult:
    """Outcome of running one piece of code against one task."""

    ok: bool
    # "pass" | "error" | "timeout" | "no_solve" | "wrong_type"
    # | "schema_mismatch" | "row_mismatch" | "blocked_import"
    status: str
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.ok


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """Wall-clock cap on a block of code.

    SIGALRM only interrupts the driver thread, so a candidate that wedges deep
    inside a JVM call can outlive this. Per-task subprocess isolation is on the
    roadmap; until then, treat the timeout as best-effort.
    """

    def _handler(signum, frame):  # noqa: ANN001
        raise TaskTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _guarded_import(name: str, *args, **kwargs):  # noqa: ANN001, ANN202
    root = name.split(".")[0]
    if root in BLOCKED_IMPORTS:
        raise BlockedImport(f"import of {root!r} is not allowed in a solution")
    return __import__(name, *args, **kwargs)


def build_fixtures(spark: SparkSession, fixtures: list[Fixture]) -> dict[str, DataFrame]:
    """Materialise every fixture as a DataFrame keyed by its declared name."""
    frames: dict[str, DataFrame] = {}
    for fx in fixtures:
        rows = [tuple(r) for r in fx.rows]
        frames[fx.name] = spark.createDataFrame(rows, schema=fx.schema)
    return frames


def _extract_solve(code: str) -> Any:
    """exec `code` and hand back its `solve` callable.

    The builtins copy is shallow but private to this call, so swapping
    __import__ here cannot leak into the host process.
    """
    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = _guarded_import
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "candidate"}

    exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102

    solve = namespace.get("solve")
    if solve is None or not callable(solve):
        raise NameError("code does not define a callable named 'solve'")
    return solve


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _normalise(value: Any, tol: float) -> Any:
    """Make a collected value comparable and hashable.

    Floats are snapped to a tolerance grid so that two mathematically equal
    results computed in different partition orders land on the same key.
    Rows/structs, arrays and maps are flattened recursively.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "__nan__"
        if math.isinf(value):
            return f"__inf_{'pos' if value > 0 else 'neg'}__"
        if tol > 0:
            return round(value / tol) * tol
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_normalise(v, tol) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _normalise(v, tol)) for k, v in value.items()))
    if hasattr(value, "asDict"):  # pyspark Row
        return tuple(
            sorted((k, _normalise(v, tol)) for k, v in value.asDict(recursive=True).items())
        )
    return value


def _sort_key(row: tuple) -> tuple:
    """Total order over heterogeneous rows, including None.

    Sorting by the raw value blows up the moment a column mixes None with an
    int, which is exactly the case null-handling tasks are built around. Keying
    on (type name, repr) is stable and never raises.
    """
    return tuple((v is None, type(v).__name__, repr(v)) for v in row)


def _schema_signature(df: DataFrame, check_order: bool) -> Any:
    pairs = [(f.name, f.dataType.simpleString()) for f in df.schema.fields]
    return pairs if check_order else sorted(pairs)


def compare_frames(
    expected: DataFrame, actual: DataFrame, cmp: Compare
) -> ExecResult:
    """Judge a candidate DataFrame against the reference DataFrame."""
    if cmp.check_schema:
        exp_sig = _schema_signature(expected, cmp.check_column_order)
        act_sig = _schema_signature(actual, cmp.check_column_order)
        if exp_sig != act_sig:
            return ExecResult(
                False,
                "schema_mismatch",
                f"expected {exp_sig}, got {act_sig}",
            )

    # Align column order before collecting so that a correct answer that simply
    # selected columns in a different order is not scored as wrong rows.
    if not cmp.check_column_order and set(expected.columns) == set(actual.columns):
        actual = actual.select(*expected.columns)

    tol = cmp.float_tolerance
    exp_rows = [tuple(_normalise(v, tol) for v in r) for r in expected.collect()]
    act_rows = [tuple(_normalise(v, tol) for v in r) for r in actual.collect()]

    if len(exp_rows) != len(act_rows):
        return ExecResult(
            False,
            "row_mismatch",
            f"expected {len(exp_rows)} rows, got {len(act_rows)}",
        )

    if cmp.mode == "rows":
        exp_rows = sorted(exp_rows, key=_sort_key)
        act_rows = sorted(act_rows, key=_sort_key)

    for i, (e, a) in enumerate(zip(exp_rows, act_rows, strict=True)):
        if e != a:
            return ExecResult(
                False,
                "row_mismatch",
                f"first difference at row {i}: expected {e!r}, got {a!r}",
            )

    return ExecResult(True, "pass")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def run_code(
    spark: SparkSession, task: Task, code: str, timeout: int = 60
) -> tuple[ExecResult, DataFrame | None]:
    """Run one candidate against one task. Never raises on candidate errors."""
    try:
        with _time_limit(timeout):
            solve = _extract_solve(code)
            frames = build_fixtures(spark, task.fixtures)
            result = solve(spark, **frames)
            if not isinstance(result, DataFrame):
                return (
                    ExecResult(
                        False,
                        "wrong_type",
                        f"solve() returned {type(result).__name__}, expected DataFrame",
                    ),
                    None,
                )
            # Force evaluation inside the time limit: Spark is lazy, so a
            # candidate that builds a broken plan would otherwise "pass" here
            # and explode later during comparison.
            result.cache()
            result.count()
            return ExecResult(True, "pass"), result
    except TaskTimeout as exc:
        return ExecResult(False, "timeout", str(exc)), None
    except BlockedImport as exc:
        return ExecResult(False, "blocked_import", str(exc)), None
    except NameError as exc:
        if "solve" in str(exc):
            return ExecResult(False, "no_solve", str(exc)), None
        return ExecResult(False, "error", f"{type(exc).__name__}: {exc}"), None
    except Exception as exc:  # noqa: BLE001 - candidate code, anything goes
        detail = str(exc).strip().splitlines()
        head = detail[0] if detail else ""
        return ExecResult(False, "error", f"{type(exc).__name__}: {head[:400]}"), None


def evaluate_candidate(
    spark: SparkSession, task: Task, code: str, timeout: int = 60
) -> ExecResult:
    """Full pipeline for one candidate: run reference, run candidate, compare."""
    ref_result, expected = run_code(spark, task, task.solution, timeout)
    if not ref_result.ok or expected is None:
        # This is a bug in the benchmark, not in the model. Surface it loudly.
        return ExecResult(
            False,
            "reference_broken",
            f"task {task.id}: reference solution failed: {ref_result.detail}",
        )

    cand_result, actual = run_code(spark, task, code, timeout)
    if not cand_result.ok or actual is None:
        return cand_result

    return compare_frames(expected, actual, task.compare)
