"""Turning a task into a model prompt, and model output back into code.

Extraction is deliberately forgiving. A model that writes correct PySpark but
wraps it in prose should score as correct -- otherwise the benchmark is partly
measuring instruction-following formatting, which is a different axis and one
that would flatter models tuned on this exact style.
"""

from __future__ import annotations

import re

from .schema import Task

SYSTEM_PROMPT = (
    "You are an expert PySpark engineer. You write correct, idiomatic PySpark "
    "using the DataFrame API. You respond with a single Python code block and "
    "no explanation."
)

_TEMPLATE = """{prompt}

Input DataFrames (already created, passed as arguments):
{fixtures}

Write a single Python function with exactly this signature:

    def solve(spark, {args}):
        ...
        return result

Requirements:
- Return a PySpark DataFrame.
- Put any imports you need inside the code block (e.g. `from pyspark.sql import functions as F`).
- Do not create your own data. Use only the DataFrames passed in.
- Do not call .show(), .collect(), or print().
"""


def describe_fixtures(task: Task) -> str:
    lines = []
    for fx in task.fixtures:
        lines.append(f"  {fx.name}: {fx.schema}")
    return "\n".join(lines)


def build_prompt(task: Task) -> str:
    """The user-turn text for a task."""
    return _TEMPLATE.format(
        prompt=task.prompt,
        fixtures=describe_fixtures(task),
        args=", ".join(task.fixture_names),
    )


_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull runnable Python out of a model response.

    Handles, in order: reasoning-model <think> blocks, fenced code blocks
    (picking the one that actually defines solve), and bare code.
    """
    if not text:
        return ""

    # Reasoning traces routinely contain draft code that does not run. Strip
    # them before looking for the answer, or we score the scratchpad.
    text = _THINK_RE.sub("", text)
    # An unterminated <think> means the model ran out of budget mid-reasoning.
    if "<think>" in text.lower():
        text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)

    blocks = [b.strip() for b in _FENCE_RE.findall(text)]
    if blocks:
        for block in blocks:
            if "def solve(" in block:
                return block
        return blocks[0]

    if "def solve(" in text:
        # Bare code, no fence. Drop any prose before the first import/def.
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.match(r"^\s*(from|import|def)\s", line):
                return "\n".join(lines[i:]).strip()

    return text.strip()
