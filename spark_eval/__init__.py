"""spark-eval: an execution-based benchmark for PySpark code generation."""

from .harness import ExecResult, compare_frames, evaluate_candidate, run_code
from .prompting import build_prompt, extract_code
from .runner import RunReport, format_report, get_spark, pass_at_k, run_suite
from .schema import Task, TaskValidationError, load_task, load_tasks

__version__ = "0.1.0"

__all__ = [
    "ExecResult",
    "RunReport",
    "Task",
    "TaskValidationError",
    "build_prompt",
    "compare_frames",
    "evaluate_candidate",
    "extract_code",
    "format_report",
    "get_spark",
    "load_task",
    "load_tasks",
    "pass_at_k",
    "run_code",
    "run_suite",
]
