"""Run a model over the suite and score it.

pass@k uses the unbiased estimator from Chen et al. (2021), "Evaluating Large
Language Models Trained on Code" -- not the naive "did any of k samples pass",
which is biased upward and not comparable across different n.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession

from .harness import ExecResult, evaluate_candidate
from .models import DummyModel, Model, ModelError
from .prompting import build_prompt, extract_code
from .schema import Task


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples drawn from n passes.

    n = samples generated, c = samples that passed.
    """
    if n < k:
        raise ValueError(f"cannot estimate pass@{k} from only {n} samples")
    if n - c < k:
        return 1.0
    # Product form avoids overflow in the binomial coefficients.
    prob = 1.0
    for i in range(k):
        prob *= (n - c - i) / (n - i)
    return 1.0 - prob


@dataclass
class SampleRecord:
    task_id: str
    category: str
    difficulty: str
    sample_index: int
    passed: bool
    status: str
    detail: str
    raw_response: str
    extracted_code: str
    latency_s: float


@dataclass
class TaskRecord:
    task_id: str
    category: str
    difficulty: str
    probes: str
    n: int
    c: int
    statuses: dict[str, int] = field(default_factory=dict)


@dataclass
class RunReport:
    model: str
    n_samples: int
    temperature: float
    started_at: str
    duration_s: float
    n_tasks: int
    pass_at_1: float
    pass_at_k: dict[str, float]
    by_category: dict[str, dict]
    by_difficulty: dict[str, dict]
    failure_modes: dict[str, int]
    tasks: list[TaskRecord]
    samples: list[SampleRecord]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def get_spark(app_name: str = "spark-eval") -> SparkSession:
    """A small, deterministic, local Spark session.

    Single shuffle partition is deliberate: it makes float aggregation order
    reproducible and cuts per-task overhead by more than half. Tasks are tiny;
    parallelism buys nothing here.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def run_suite(
    tasks: list[Task],
    model: Model,
    *,
    n_samples: int = 1,
    ks: tuple[int, ...] = (1,),
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 60,
    spark: SparkSession | None = None,
    keep_responses: bool = True,
    progress: bool = True,
) -> RunReport:
    owns_spark = spark is None
    spark = spark or get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    started = datetime.now(timezone.utc)
    t0 = time.time()

    samples: list[SampleRecord] = []
    task_records: list[TaskRecord] = []
    failure_modes: dict[str, int] = defaultdict(int)

    for idx, task in enumerate(tasks, 1):
        prompt = build_prompt(task)
        # The dummy backend needs the gold answer keyed by the exact prompt.
        if isinstance(model, DummyModel):
            model.register(prompt, task.solution)

        statuses: dict[str, int] = defaultdict(int)
        passed_count = 0

        for s in range(n_samples):
            s_t0 = time.time()
            try:
                raw = model.generate(prompt, temperature, max_tokens)
            except ModelError as exc:
                raw = ""
                result = ExecResult(False, "model_error", str(exc))
                code = ""
            else:
                code = extract_code(raw)
                if not code.strip():
                    result = ExecResult(False, "empty_response", "no code in response")
                else:
                    result = evaluate_candidate(spark, task, code, timeout=timeout)

            latency = time.time() - s_t0
            statuses[result.status] += 1
            if result.ok:
                passed_count += 1
            else:
                failure_modes[result.status] += 1

            samples.append(
                SampleRecord(
                    task_id=task.id,
                    category=task.category,
                    difficulty=task.difficulty,
                    sample_index=s,
                    passed=result.ok,
                    status=result.status,
                    detail=result.detail,
                    raw_response=raw if keep_responses else "",
                    extracted_code=code if keep_responses else "",
                    latency_s=round(latency, 3),
                )
            )

            # A broken reference solution means the benchmark is lying. Stop.
            if result.status == "reference_broken":
                raise RuntimeError(result.detail)

        task_records.append(
            TaskRecord(
                task_id=task.id,
                category=task.category,
                difficulty=task.difficulty,
                probes=task.probes,
                n=n_samples,
                c=passed_count,
                statuses=dict(statuses),
            )
        )

        if progress:
            mark = "PASS" if passed_count == n_samples else (
                "FAIL" if passed_count == 0 else f"{passed_count}/{n_samples}"
            )
            print(
                f"[{idx:>3}/{len(tasks)}] {task.id:<40} {mark}",
                flush=True,
            )

    def _agg(records: list[TaskRecord]) -> dict:
        if not records:
            return {"n_tasks": 0, "pass_at_1": 0.0}
        out = {
            "n_tasks": len(records),
            "pass_at_1": round(
                sum(pass_at_k(r.n, r.c, 1) for r in records) / len(records), 4
            ),
        }
        for k in ks:
            if k > 1 and n_samples >= k:
                out[f"pass_at_{k}"] = round(
                    sum(pass_at_k(r.n, r.c, k) for r in records) / len(records), 4
                )
        return out

    by_category: dict[str, dict] = {}
    for cat in sorted({r.category for r in task_records}):
        by_category[cat] = _agg([r for r in task_records if r.category == cat])

    by_difficulty: dict[str, dict] = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in task_records if r.difficulty == diff]
        if subset:
            by_difficulty[diff] = _agg(subset)

    overall = _agg(task_records)
    report = RunReport(
        model=model.name,
        n_samples=n_samples,
        temperature=temperature,
        started_at=started.isoformat(),
        duration_s=round(time.time() - t0, 2),
        n_tasks=len(task_records),
        pass_at_1=overall["pass_at_1"],
        pass_at_k={
            f"pass_at_{k}": overall[f"pass_at_{k}"]
            for k in ks
            if k > 1 and f"pass_at_{k}" in overall
        },
        by_category=by_category,
        by_difficulty=by_difficulty,
        failure_modes=dict(sorted(failure_modes.items(), key=lambda kv: -kv[1])),
        tasks=task_records,
        samples=samples,
    )

    if owns_spark:
        spark.stop()

    return report


def format_report(report: RunReport) -> str:
    """Human-readable summary. The JSON is the machine-readable artifact."""
    lines = [
        "",
        "=" * 68,
        f"  spark-eval  |  {report.model}",
        "=" * 68,
        f"  tasks       {report.n_tasks}",
        f"  samples     {report.n_samples} per task @ temperature {report.temperature}",
        f"  duration    {report.duration_s}s",
        "",
        f"  pass@1      {report.pass_at_1:.1%}",
    ]
    for key, v in report.pass_at_k.items():
        label = "pass@" + key.rsplit("_", 1)[-1]
        lines.append(f"  {label:<11} {v:.1%}")

    def _plural(n: int) -> str:
        return f"{n} task" if n == 1 else f"{n} tasks"

    # Worst category first: the point of the breakdown is finding the weakness.
    lines += ["", "  By category", "  " + "-" * 46]
    for cat, stats in sorted(
        report.by_category.items(), key=lambda kv: kv[1]["pass_at_1"]
    ):
        lines.append(
            f"    {cat:<22} {stats['pass_at_1']:>6.1%}  ({_plural(stats['n_tasks'])})"
        )

    if report.by_difficulty:
        lines += ["", "  By difficulty", "  " + "-" * 46]
        for diff, stats in report.by_difficulty.items():
            lines.append(
                f"    {diff:<22} {stats['pass_at_1']:>6.1%}  ({_plural(stats['n_tasks'])})"
            )

    if report.failure_modes:
        lines += ["", "  Failure modes", "  " + "-" * 46]
        total = sum(report.failure_modes.values())
        for mode, count in report.failure_modes.items():
            lines.append(f"    {mode:<22} {count:>4}  ({count / total:.0%})")

    lines += ["", "=" * 68, ""]
    return "\n".join(lines)


def write_report(report: RunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json())
