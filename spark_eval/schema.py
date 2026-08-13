"""Task schema, loading, and validation.

A task is a YAML file. The contract is deliberately narrow so that a task is
cheap to write by hand and impossible to score ambiguously:

  - `fixtures` declare the input DataFrames by schema + literal rows. They are
    small, deterministic, and committed to the repo. No network, no generated
    data, no randomness.
  - `prompt` is what the model sees. It names the fixtures and states the
    required entrypoint signature.
  - `solution` is reference PySpark that a human wrote and that the harness
    executes to produce the expected output. There is no hardcoded expected
    table anywhere -- expected output is *computed*, so a fixture edit can
    never silently desynchronise from a stale golden file.
  - `compare` says how to judge equality. Default is order-insensitive rows
    plus exact schema.

Every task must define `solve(spark, **frames) -> DataFrame`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CATEGORIES = {
    "joins",
    "windows",
    "aggregations",
    "schema_nested",
    "udf_vs_native",
    "nulls_types",
    "sql_translation",
    "delta_merge",
}

DIFFICULTIES = {"easy", "medium", "hard"}

_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class TaskValidationError(ValueError):
    """Raised when a task file is structurally invalid."""


@dataclass(frozen=True)
class Fixture:
    """One input DataFrame, defined literally.

    `schema` is a Spark DDL string (e.g. "id INT, name STRING"). We use DDL
    rather than inferring from rows because inference silently changes types
    when a column happens to be all-null in the sample, and null handling is
    one of the things this benchmark is trying to measure.
    """

    name: str
    schema: str
    rows: list[list[Any]]

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise TaskValidationError(
                f"fixture name {self.name!r} is not a valid Python identifier"
            )
        if not self.schema.strip():
            raise TaskValidationError(f"fixture {self.name!r} has an empty schema")


@dataclass(frozen=True)
class Compare:
    """How to decide whether a candidate result matches the reference."""

    # "rows" -> order-insensitive multiset comparison (the default; most tasks
    #           do not specify an order, so requiring one would fail correct code)
    # "ordered_rows" -> order matters (use when the prompt explicitly asks for
    #           a sort, e.g. window/top-n tasks)
    mode: Literal["rows", "ordered_rows"] = "rows"

    # Exact schema match (names, types, nullability-insensitive). Turning this
    # off is a deliberate loosening -- record why in the task file.
    check_schema: bool = True

    # Column names must match exactly and in order. Off means we compare on the
    # set of columns, useful when the prompt does not pin an output column order.
    check_column_order: bool = False

    # Absolute tolerance for float/double columns. Spark's floating point
    # aggregation order is not deterministic across partitions, so exact
    # equality on doubles is a flaky-test generator.
    float_tolerance: float = 1e-9


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    difficulty: str
    prompt: str
    fixtures: list[Fixture]
    solution: str
    compare: Compare = field(default_factory=Compare)
    # Free-text note on what this task is actually probing. Shows up in the
    # per-category failure report; the point of the benchmark is diagnosis,
    # not just a number.
    probes: str = ""
    tags: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def fixture_names(self) -> list[str]:
        return [f.name for f in self.fixtures]


def _require(data: dict, key: str, path: Path, type_: type) -> Any:
    if key not in data:
        raise TaskValidationError(f"{path}: missing required key {key!r}")
    value = data[key]
    if not isinstance(value, type_):
        raise TaskValidationError(
            f"{path}: key {key!r} must be {type_.__name__}, got {type(value).__name__}"
        )
    return value


def load_task(path: Path) -> Task:
    """Parse and validate a single task file."""
    with path.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise TaskValidationError(f"{path}: top level must be a mapping")

    task_id = _require(raw, "id", path, str)
    if not _ID_RE.match(task_id):
        raise TaskValidationError(
            f"{path}: id {task_id!r} must be lower_snake_case"
        )

    category = _require(raw, "category", path, str)
    if category not in CATEGORIES:
        raise TaskValidationError(
            f"{path}: unknown category {category!r}; expected one of {sorted(CATEGORIES)}"
        )

    difficulty = raw.get("difficulty", "medium")
    if difficulty not in DIFFICULTIES:
        raise TaskValidationError(
            f"{path}: difficulty {difficulty!r} must be one of {sorted(DIFFICULTIES)}"
        )

    prompt = _require(raw, "prompt", path, str).strip()
    solution = _require(raw, "solution", path, str)

    raw_fixtures = _require(raw, "fixtures", path, list)
    if not raw_fixtures:
        raise TaskValidationError(f"{path}: at least one fixture is required")

    fixtures = []
    for item in raw_fixtures:
        if not isinstance(item, dict):
            raise TaskValidationError(f"{path}: each fixture must be a mapping")
        fixtures.append(
            Fixture(
                name=_require(item, "name", path, str),
                schema=_require(item, "schema", path, str),
                rows=[list(r) for r in _require(item, "rows", path, list)],
            )
        )

    names = [f.name for f in fixtures]
    if len(set(names)) != len(names):
        raise TaskValidationError(f"{path}: duplicate fixture names in {names}")

    raw_compare = raw.get("compare") or {}
    if not isinstance(raw_compare, dict):
        raise TaskValidationError(f"{path}: 'compare' must be a mapping")
    unknown = set(raw_compare) - {
        "mode",
        "check_schema",
        "check_column_order",
        "float_tolerance",
    }
    if unknown:
        raise TaskValidationError(f"{path}: unknown compare keys {sorted(unknown)}")
    compare = Compare(**raw_compare)
    if compare.mode not in ("rows", "ordered_rows"):
        raise TaskValidationError(f"{path}: invalid compare.mode {compare.mode!r}")

    # The reference solution has to honour the same contract we ask of models.
    if "def solve(" not in solution:
        raise TaskValidationError(
            f"{path}: solution must define solve(spark, ...); "
            "the harness calls it by name"
        )

    # A prompt that does not mention a fixture is a prompt the model cannot
    # answer. This has caught more authoring bugs than any other check.
    for name in names:
        if name not in prompt:
            raise TaskValidationError(
                f"{path}: fixture {name!r} is never mentioned in the prompt"
            )

    return Task(
        id=task_id,
        category=category,
        difficulty=difficulty,
        prompt=prompt,
        fixtures=fixtures,
        solution=solution,
        compare=compare,
        probes=raw.get("probes", ""),
        tags=list(raw.get("tags", [])),
        source_path=path,
    )


def load_tasks(
    root: Path,
    categories: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[Task]:
    """Load every task under `root`, optionally filtered.

    Sorted by id so that runs are reproducible and diffable.
    """
    paths = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    tasks = [load_task(p) for p in paths]

    seen: dict[str, Path] = {}
    for t in tasks:
        if t.id in seen:
            raise TaskValidationError(
                f"duplicate task id {t.id!r} in {t.source_path} and {seen[t.id]}"
            )
        seen[t.id] = t.source_path  # type: ignore[assignment]

    if categories:
        tasks = [t for t in tasks if t.category in set(categories)]
    if ids:
        tasks = [t for t in tasks if t.id in set(ids)]

    return sorted(tasks, key=lambda t: t.id)
