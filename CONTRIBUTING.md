# Contributing a task

The bar is not "a PySpark question." It is **a question where fluent-looking code
is wrong.** If a competent model gets it right by pattern-matching, it costs a
Spark session to run and tells us nothing.

## The test for a good task

Before writing anything, answer this: *what is the specific wrong version, and
why would a strong model produce it?*

If you cannot name the mutant, the task is not ready. Good tasks come in pairs —
a correct implementation and a plausible wrong one that differ by one method
name.

Examples of the shape we want:

| Correct | Plausible mutant | Why the mutant is tempting |
|---|---|---|
| `explode_outer` | `explode` | reads as a synonym; only differs on empty/null arrays |
| default RANGE frame | `rowsBetween(unboundedPreceding, 0)` | looks more explicit and "safer" |
| `eqNullSafe` | `==` | `==` is what a join condition normally looks like |
| `coalesce(sum(x), 0)` | `sum(coalesce(x, 0))` | reads identically in English |

Examples of what to skip: "read a Parquet file", "filter rows where x > 5",
"rename a column". Correctness there is not in question.

## Writing it

1. Create `tasks/<category>/<id>.yaml`. The id must be `lower_snake_case` and
   match the filename.
2. Fill in `probes` — one or two sentences on the semantic trap. This is not
   decoration; it is what turns a failure report into a diagnosis, and it shows
   up in the per-task output.
3. Keep fixtures **small and adversarial**. Four to eight rows is usually
   enough. Every row should exist for a reason: a NULL, a tie, a duplicate key,
   an empty array. A fixture row that does not discriminate between correct and
   incorrect code is dead weight.
4. Mention every fixture by name in the prompt — the loader enforces this,
   because a fixture the prompt never mentions is one the model cannot use.
5. Write the reference solution as `solve(spark, <fixture names>) -> DataFrame`.
   Comment the line that carries the semantics.
6. Set `compare.mode: ordered_rows` **only** if the prompt explicitly asks for a
   sort. Otherwise a correct answer in a different order fails.
7. Use `float_tolerance` on any task that sums or averages doubles.

## Adding the mutant (required)

Add an entry to `MUTANTS` in `tests/test_mutants.py`:

```python
"your_task_id": (
    "one line on what the mistake is",
    """
from pyspark.sql import functions as F

def solve(spark, df):
    ...   # the wrong version
""",
),
```

`test_every_task_has_a_mutant` fails CI if you skip this.

If your mutant *passes*, the task cannot distinguish correct from incorrect code.
That is a bug in the fixtures, not in the harness — add the row that separates
them. This happens more often than you would expect, and catching it is the whole
reason the mutant tests exist.

## Before you open the PR

```bash
spark-eval validate                    # structural checks
spark-eval selfcheck --id your_task_id # the reference actually runs
pytest -q                              # mutants rejected, units green
```

## Things we will push back on

- **Generated fixtures or assertions.** A benchmark whose assertions came from an
  LLM measures the generator. Hand-write them.
- **Tasks with more than one defensible answer.** If two reasonable engineers
  would produce different DataFrames, the task is underspecified — pin it in the
  prompt or drop it.
- **Anything needing network, a cluster, or a file on disk.** Fixtures are
  literal rows, committed to the repo, deterministic.
- **Difficulty inflation.** `hard` means the semantics are subtle, not that the
  query is long. A 40-line pipeline of easy steps is an `easy` task that wastes
  a minute of runtime.
