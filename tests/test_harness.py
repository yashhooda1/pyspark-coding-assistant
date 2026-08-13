"""Unit tests for the comparator and the execution guards.

The comparator is the thing everything else trusts, so its edge cases get
tested directly rather than only through tasks.
"""

from __future__ import annotations

import pytest

from spark_eval.harness import compare_frames, run_code
from spark_eval.runner import pass_at_k
from spark_eval.schema import Compare, Fixture, Task


def _frame(spark, schema, rows):
    return spark.createDataFrame([tuple(r) for r in rows], schema=schema)


def _task(solution: str, compare: Compare | None = None) -> Task:
    return Task(
        id="t",
        category="joins",
        difficulty="easy",
        prompt="uses df",
        fixtures=[Fixture(name="df", schema="a INT", rows=[[1], [2]])],
        solution=solution,
        compare=compare or Compare(),
    )


# --------------------------------------------------------------------------
# Comparison semantics
# --------------------------------------------------------------------------


def test_row_order_ignored_by_default(spark):
    a = _frame(spark, "x INT", [[1], [2], [3]])
    b = _frame(spark, "x INT", [[3], [1], [2]])
    assert compare_frames(a, b, Compare(mode="rows")).ok


def test_row_order_enforced_when_requested(spark):
    a = _frame(spark, "x INT", [[1], [2], [3]])
    b = _frame(spark, "x INT", [[3], [1], [2]])
    assert not compare_frames(a, b, Compare(mode="ordered_rows")).ok


def test_nulls_sort_without_raising(spark):
    """A column mixing NULL and INT used to blow up the sort key."""
    a = _frame(spark, "x INT", [[1], [None], [3]])
    b = _frame(spark, "x INT", [[None], [3], [1]])
    assert compare_frames(a, b, Compare(mode="rows")).ok


def test_null_is_not_equal_to_zero(spark):
    a = _frame(spark, "x INT", [[None]])
    b = _frame(spark, "x INT", [[0]])
    assert not compare_frames(a, b, Compare()).ok


def test_schema_mismatch_detected(spark):
    a = _frame(spark, "x INT", [[1]])
    b = _frame(spark, "x BIGINT", [[1]])
    result = compare_frames(a, b, Compare(check_schema=True))
    assert not result.ok
    assert result.status == "schema_mismatch"


def test_schema_check_can_be_relaxed(spark):
    a = _frame(spark, "x INT", [[1]])
    b = _frame(spark, "x BIGINT", [[1]])
    assert compare_frames(a, b, Compare(check_schema=False)).ok


def test_column_order_ignored_by_default(spark):
    a = _frame(spark, "x INT, y INT", [[1, 2]])
    b = _frame(spark, "y INT, x INT", [[2, 1]])
    assert compare_frames(a, b, Compare()).ok


def test_column_order_enforced_when_requested(spark):
    a = _frame(spark, "x INT, y INT", [[1, 2]])
    b = _frame(spark, "y INT, x INT", [[2, 1]])
    assert not compare_frames(a, b, Compare(check_column_order=True)).ok


def test_float_tolerance_absorbs_partition_order(spark):
    a = _frame(spark, "x DOUBLE", [[0.1 + 0.2]])
    b = _frame(spark, "x DOUBLE", [[0.3]])
    assert compare_frames(a, b, Compare(float_tolerance=1e-6)).ok
    assert not compare_frames(a, b, Compare(float_tolerance=0.0)).ok


def test_row_count_mismatch_reported(spark):
    a = _frame(spark, "x INT", [[1], [2]])
    b = _frame(spark, "x INT", [[1]])
    result = compare_frames(a, b, Compare())
    assert result.status == "row_mismatch"
    assert "expected 2 rows, got 1" in result.detail


def test_duplicate_rows_are_significant(spark):
    """Multiset, not set: a dropDuplicates bug must not pass."""
    a = _frame(spark, "x INT", [[1], [1], [2]])
    b = _frame(spark, "x INT", [[1], [2], [2]])
    assert not compare_frames(a, b, Compare()).ok


def test_nested_struct_compared_recursively(spark):
    schema = "s STRUCT<a: INT, b: STRING>"
    a = _frame(spark, schema, [[(1, "x")]])
    b = _frame(spark, schema, [[(1, "y")]])
    assert not compare_frames(a, b, Compare()).ok
    assert compare_frames(a, a, Compare()).ok


def test_array_order_is_significant(spark):
    a = _frame(spark, "xs ARRAY<INT>", [[[1, 2]]])
    b = _frame(spark, "xs ARRAY<INT>", [[[2, 1]]])
    assert not compare_frames(a, b, Compare()).ok


# --------------------------------------------------------------------------
# Execution guards
# --------------------------------------------------------------------------


def test_missing_solve_reported_cleanly(spark):
    result, _ = run_code(spark, _task("def other(): pass"), "def other(): pass")
    assert result.status == "no_solve"


def test_non_dataframe_return_reported(spark):
    code = "def solve(spark, df):\n    return 42\n"
    result, _ = run_code(spark, _task(code), code)
    assert result.status == "wrong_type"


def test_syntax_error_is_a_failure_not_a_crash(spark):
    code = "def solve(spark, df:\n    return df\n"
    result, _ = run_code(spark, _task(code), code)
    assert result.status == "error"
    assert "SyntaxError" in result.detail


def test_blocked_import_is_rejected(spark):
    code = "import os\n\ndef solve(spark, df):\n    return df\n"
    result, _ = run_code(spark, _task(code), code)
    assert result.status == "blocked_import"


def test_lazy_failure_is_caught_at_run_time(spark):
    """Spark is lazy; a broken plan must fail here, not later in compare."""
    code = (
        "from pyspark.sql import functions as F\n"
        "def solve(spark, df):\n"
        "    return df.select(F.col('does_not_exist'))\n"
    )
    result, _ = run_code(spark, _task(code), code)
    assert result.status == "error"


def test_timeout_is_enforced(spark):
    code = (
        "def solve(spark, df):\n"
        "    while True:\n"
        "        pass\n"
    )
    result, _ = run_code(spark, _task(code), code, timeout=2)
    assert result.status == "timeout"


# --------------------------------------------------------------------------
# pass@k estimator
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (1, 1, 1, 1.0),
        (1, 0, 1, 0.0),
        (10, 0, 1, 0.0),
        (10, 10, 5, 1.0),
        (10, 5, 1, 0.5),
        # 2 of 4 pass; drawing 2 of 4 misses both only 1 time in 6.
        (4, 2, 2, 1 - (2 / 4) * (1 / 3)),
    ],
)
def test_pass_at_k_values(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_rejects_k_greater_than_n():
    with pytest.raises(ValueError, match="cannot estimate"):
        pass_at_k(2, 1, 5)


def test_pass_at_k_is_monotonic_in_k():
    values = [pass_at_k(10, 3, k) for k in range(1, 8)]
    assert values == sorted(values)
