# spark-eval

**An execution-based benchmark for PySpark code generation.**

Every task runs the generated code against a real Spark session and compares the
resulting DataFrame to a reference. Nothing here scores string similarity.

```bash
pip install -e .
spark-eval selfcheck                          # every reference solution executes
spark-eval run --model ollama:qwen3:4b        # score a local model
spark-eval run --model openai:gpt-4o-mini --n 5 --k 5 --out report.json
```

---

## Why this exists

I fine-tuned a PySpark coding assistant and evaluated it the way most people do:
token accuracy, ROUGE, and an LLM judge. It scored **0.833 mean token accuracy**
and a validation loss of 0.757. By those numbers it worked.

Then I ran the outputs. **One in three was correct.**

The model had learned what PySpark *looks like* — the right imports, plausible
method chains, idiomatic naming — without learning what the operators *do*. No
text-similarity metric can see that gap, because the wrong answer and the right
answer differ by one method name and look equally fluent. An LLM judge does not
reliably catch it either: judges score fluent, well-structured code generously,
and `rowsBetween` versus the default RANGE frame is exactly the kind of
difference a judge waves through.

The only thing that catches it is running the code.

There are execution-based benchmarks for general Python (HumanEval, MBPP) and for
data science on single-node libraries — DS-1000 covers NumPy, Pandas, SciPy,
scikit-learn, PyTorch, TensorFlow and Matplotlib. **None of them cover PySpark.**
That is a strange hole: Spark is the backbone of most enterprise data platforms,
and those are the platforms currently having LLMs pointed at them.

spark-eval fills it.

## What makes Spark different from pandas

You cannot port a pandas benchmark and call it done. Spark's semantics diverge
in ways that produce *silently wrong* results rather than exceptions:

- **NULL is not a value.** `NULL = NULL` is unknown, not true. This means a left-anti
  join keeps NULL-keyed rows, an inner join drops them, and matching NULL to NULL
  needs `eqNullSafe`. Same operator, three different correct behaviours.
- **Window frames default to RANGE, not ROWS.** An ordered window with no explicit
  frame uses `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. With tied
  ordering keys that is a different answer from the `rowsBetween` version people
  write. Both look correct in review.
- **Aggregates swallow NULLs, then return one.** `sum()` skips NULLs, but a group
  where everything is NULL aggregates to NULL, not 0.
- **Laziness hides errors.** A broken plan builds fine and fails on the action —
  so a harness that does not force evaluation will score broken code as working.
- **Joins fan out.** Non-unique keys multiply rows, and any aggregate downstream
  silently changes meaning.

Each of those is a category in this suite, chosen because it is a place where
fluent-looking code is wrong.

## How a task works

A task is one YAML file:

```yaml
id: window_default_frame_ties
category: windows
difficulty: hard
probes: >
  With an ORDER BY and no explicit frame, the default is RANGE BETWEEN
  UNBOUNDED PRECEDING AND CURRENT ROW -- a value range, so tied keys see
  each other's rows. rowsBetween(...) gives a different, wrong answer.

prompt: |
  For each row in `events`, compute a running total of value within each
  user, ordered by ts, using the SQL default window frame. Rows sharing a
  ts within a user must therefore share the same running total.

fixtures:
  - name: events
    schema: user STRING, ts INT, value INT
    rows:
      - ["a", 1, 10]
      - ["a", 2, 20]
      - ["a", 2, 30]

solution: |
  from pyspark.sql import functions as F
  from pyspark.sql.window import Window

  def solve(spark, events):
      w = Window.partitionBy("user").orderBy("ts")
      return events.withColumn("running", F.sum("value").over(w))

compare:
  mode: rows
```

The model is shown the prompt and the fixture schemas, and asked for
`solve(spark, **frames) -> DataFrame`. The harness builds the fixtures, runs the
reference to get the expected DataFrame, runs the candidate, and compares.

**Expected output is computed, never stored.** There is no golden file to drift
out of sync when a fixture is edited.

### Design decisions worth arguing with

- **Order-insensitive by default.** Most prompts do not pin an ordering, so
  requiring one would fail correct code. Tasks that *do* ask for a sort set
  `mode: ordered_rows`.
- **Multiset, not set.** Duplicate rows matter — otherwise a stray
  `dropDuplicates()` passes.
- **Float tolerance.** Spark's floating-point aggregation order varies with
  partitioning. Exact double equality is a flaky-test generator.
- **Column order ignored by default.** Selecting the right columns in a
  different order is not a wrong answer unless the prompt said so.
- **Schema checked by default.** Returning `BIGINT` where `INT` was asked for is
  a real bug in a pipeline that writes to a typed table.
- **Forgiving extraction.** Prose around the code, missing fences, and
  `<think>` blocks are all handled. The benchmark measures Spark semantics, not
  formatting compliance — otherwise it would quietly favour models tuned on this
  exact response style.

## Categories

| Category | Tasks | What it probes |
|---|---|---|
| `joins` | 2 | fanout, anti-join NULL semantics, match vs. NULL value |
| `windows` | 2 | RANGE vs ROWS frames, rank family under ties |
| `aggregations` | 2 | count/countDistinct NULL semantics, pivot |
| `nulls_types` | 2 | null-safe equality, all-NULL groups, coercion |
| `schema_nested` | 1 | explode vs explode_outer, structs, arrays |
| `udf_vs_native` | 1 | UDF NULL handling, return types |
| `sql_translation` | 1 | WHERE vs HAVING, clause ordering |
| `delta_merge` | 1 | upsert / latest-version-wins semantics |

**12 tasks today.** This is a seed set, not the finished benchmark — it exists to
prove the harness and fix the shape of a task. Target is 150–200, hand-written.
See [CONTRIBUTING.md](CONTRIBUTING.md).

Deliberately: 150 good tasks beat 1,000 generated ones. The fixtures and
assertions here are written by hand, because a benchmark whose assertions were
LLM-generated measures the generator, not the model under test.

## Is the benchmark itself correct?

Two mechanisms, both enforced in CI.

**`spark-eval selfcheck`** executes every reference solution and compares it to
itself. A task whose gold answer does not run produces meaningless scores for
every model, so this is a hard gate.

**Mutation tests** are the real check. Every task ships with a *mutant* — the
specific wrong implementation a model actually tends to produce — and the suite
asserts the harness rejects it:

| Task | Mutant that must fail |
|---|---|
| `join_anti_null_key` | treats anti-join as set difference, dropping NULL keys |
| `window_default_frame_ties` | explicit ROWS frame instead of default RANGE |
| `null_safe_equality_join` | plain `==`, dropping the NULL/NULL pair |
| `nested_explode_outer_empty` | `explode` instead of `explode_outer` |
| `agg_count_null_semantics` | `count("*")` for all three counts |
| `upsert_latest_version` | overwrite instead of upsert, losing target-only rows |
| … | one per task, enforced by `test_every_task_has_a_mutant` |

Without this, a comparator that always returned `True` would pass selfcheck
perfectly. `pytest` currently runs **50 tests**: 12 mutants, the pairing check,
and unit tests covering NULL sorting, nested structs, multiset semantics, lazy
evaluation, timeouts, the import guard, and the pass@k estimator.

## Scoring

`pass@k` uses the unbiased estimator from Chen et al. (2021), not "did any of k
samples pass" — the naive version is biased upward and not comparable across
different `n`.

Reports break down by category, by difficulty, and by **failure mode**, because
the interesting output is not the headline number:

```
  pass@1      41.7%

  By category
    windows                  0.0%  (2 tasks)
    nulls_types             50.0%  (2 tasks)
    joins                  100.0%  (2 tasks)

  Failure modes
    row_mismatch              4  (57%)
    error                     2  (29%)
    schema_mismatch           1  (14%)
```

`row_mismatch` means the model wrote runnable Spark that computed the wrong
thing. That is the number this benchmark was built to expose.

## Backends

| Spec | Backend |
|---|---|
| `ollama:qwen3:4b` | local Ollama server (`OLLAMA_HOST`) |
| `openai:gpt-4o-mini` | any OpenAI-compatible endpoint (`OPENAI_BASE_URL`) |
| `dummy:reference` | echoes gold solutions; must score 100% |
| `dummy:empty` | returns nothing; must score 0% |

The OpenAI-compatible adapter covers vLLM, llama.cpp's server, TGI, OpenRouter,
and the hosted frontier APIs. Adding a backend is one method.

## Safety

`spark-eval run` **executes untrusted model output in your Python process.** The
import guard blocks `os`, `subprocess`, `shutil`, `socket`, `pathlib` and friends,
which stops a model that decides to tidy up your filesystem. It is not a security
boundary and is not meant to be one — a determined adversary gets out of it.

If you are scoring checkpoints you did not train, run it in a container with no
network and a read-only mount.

## Roadmap

- [ ] 150–200 tasks (12 today)
- [ ] Real Delta Lake tasks behind a `[delta]` extra — the current `delta_merge`
      task tests upsert *logic* without the `delta-spark` dependency
- [ ] Subprocess isolation per task (`--isolate`)
- [ ] Published leaderboard as a Hugging Face Space
- [ ] Contamination canaries to detect training on the suite

## Related work

- **HumanEval** / **MBPP** — general Python, execution-based. No data-engineering surface.
- **[DS-1000](https://proceedings.mlr.press/v202/lai23b/lai23b.pdf)** — the closest
  relative: execution-based, data science, seven libraries. No Spark.
- **[CodeBenchGen](https://arxiv.org/pdf/2404.00566v2)** — generates execution
  sandboxes automatically. Complementary; spark-eval hand-writes assertions
  because the failure modes it targets are too subtle to generate reliably.

## License

Apache-2.0.

Built by [Yash Hooda](https://www.yashhooda.ai/) — [Hugging Face](https://huggingface.co/hoodarunner) · [Ollama](https://ollama.com/hoodarunner)
