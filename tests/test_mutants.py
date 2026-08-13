"""Mutation tests: prove the harness fails plausible-but-wrong code.

A benchmark that only checks "the gold solution passes" is worthless -- an
always-return-True comparator would satisfy that. Every task here gets a
*mutant*: the specific wrong implementation a model actually tends to produce.
The suite asserts the harness rejects it.

If you add a task, add its mutant. CI enforces the pairing.
"""

from __future__ import annotations

import pytest

from spark_eval.harness import evaluate_candidate
from spark_eval.schema import load_tasks

from .conftest import TASKS_DIR

# task_id -> (description of the mistake, wrong implementation)
MUTANTS: dict[str, tuple[str, str]] = {
    "join_anti_null_key": (
        "treats anti-join as set difference, dropping NULL-keyed rows",
        """
from pyspark.sql import functions as F

def solve(spark, orders, customers):
    ids = [r[0] for r in customers.select("customer_id").collect()]
    return orders.filter(~F.col("customer_id").isin(ids))
""",
    ),
    "join_fanout_duplicate_keys": (
        "dedups the right side first, so the fanout never happens",
        """
from pyspark.sql import functions as F

def solve(spark, sales, rates):
    r = rates.dropDuplicates(["region"])
    return (sales.join(r, on="region", how="inner")
                 .groupBy("region")
                 .agg(F.sum(F.col("amount") * F.col("multiplier")).alias("total")))
""",
    ),
    "window_default_frame_ties": (
        "uses an explicit ROWS frame instead of the default RANGE frame",
        """
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def solve(spark, events):
    w = Window.partitionBy("user").orderBy("ts").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    return events.withColumn("running", F.sum("value").over(w))
""",
    ),
    "window_rank_family_ties": (
        "uses row_number() for all three columns, erasing tie behaviour",
        """
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def solve(spark, scores):
    w = Window.partitionBy("team").orderBy(F.col("points").desc(), F.col("player").asc())
    return (scores
            .withColumn("rnk", F.row_number().over(w))
            .withColumn("dense", F.row_number().over(w))
            .withColumn("rownum", F.row_number().over(w))
            .orderBy(F.col("team").asc(), F.col("points").desc(), F.col("player").asc()))
""",
    ),
    "agg_count_null_semantics": (
        "uses count('*') everywhere, ignoring NULL and DISTINCT semantics",
        """
from pyspark.sql import functions as F

def solve(spark, staff):
    return staff.groupBy("dept").agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.count(F.lit(1)).alias("n_emails"),
        F.count(F.lit(1)).alias("n_distinct"),
    )
""",
    ),
    "null_sum_all_null_group": (
        "plain sum(), so the all-NULL group returns NULL instead of 0",
        """
from pyspark.sql import functions as F

def solve(spark, readings):
    return readings.groupBy("region").agg(F.sum("value").cast("long").alias("total"))
""",
    ),
    "null_safe_equality_join": (
        "plain equality, which silently drops the NULL/NULL pair",
        """
from pyspark.sql import functions as F

def solve(spark, left_t, right_t):
    return (left_t.join(right_t, left_t["code"] == right_t["code"], "inner")
                  .select(left_t["code"].alias("code"), "lval", "rval"))
""",
    ),
    "nested_explode_outer_empty": (
        "explode() instead of explode_outer(), dropping empty/null arrays",
        """
from pyspark.sql import functions as F

def solve(spark, docs):
    return docs.select("doc_id", F.explode("tags").alias("tag"))
""",
    ),
    "upsert_latest_version": (
        "overwrite instead of upsert: target-only rows are lost",
        """
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def solve(spark, target, updates):
    w = Window.partitionBy("id").orderBy(F.col("version").desc())
    return (updates.withColumn("_rn", F.row_number().over(w))
                   .filter(F.col("_rn") == 1)
                   .drop("_rn"))
""",
    ),
    "udf_null_input_handling": (
        "no None guard, so the UDF raises on the NULL row",
        """
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def solve(spark, notes):
    word_udf = F.udf(lambda s: len(s.split()), IntegerType())
    return notes.withColumn("n_words", word_udf(F.col("text")))
""",
    ),
    "sql_having_to_dataframe": (
        "applies HAVING as a WHERE, filtering rows instead of groups",
        """
from pyspark.sql import functions as F

def solve(spark, txns):
    return (txns
            .filter(F.col("status") == "ok")
            .groupBy("store")
            .agg(F.sum("amount").alias("total"))
            .orderBy(F.col("total").desc()))
""",
    ),
    "agg_pivot_fill": (
        "forgets fillna, leaving NULL for absent store/quarter combinations",
        """
from pyspark.sql import functions as F

def solve(spark, sales_long):
    return (sales_long.groupBy("store")
                      .pivot("quarter", ["Q1", "Q2", "Q3"])
                      .agg(F.sum("amount")))
""",
    ),
}

ALL_TASKS = {t.id: t for t in load_tasks(TASKS_DIR)}


def test_every_task_has_a_mutant():
    """Keeps the two files honest with each other."""
    missing = sorted(set(ALL_TASKS) - set(MUTANTS))
    orphaned = sorted(set(MUTANTS) - set(ALL_TASKS))
    assert not missing, f"tasks with no mutant test: {missing}"
    assert not orphaned, f"mutants for tasks that no longer exist: {orphaned}"


@pytest.mark.parametrize("task_id", sorted(MUTANTS))
def test_mutant_is_rejected(spark, task_id):
    task = ALL_TASKS[task_id]
    description, code = MUTANTS[task_id]
    result = evaluate_candidate(spark, task, code, timeout=90)

    assert result.status != "reference_broken", (
        f"{task_id}: the reference solution itself failed -- fix the task, "
        f"not the mutant ({result.detail})"
    )
    assert not result.ok, (
        f"{task_id}: harness ACCEPTED a wrong answer ({description}). "
        f"The task cannot distinguish correct from incorrect code; "
        f"strengthen the fixtures."
    )
