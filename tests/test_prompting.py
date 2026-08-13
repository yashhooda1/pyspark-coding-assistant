"""Extraction tests.

Every case here is a real response shape seen from a local model. Extraction
bugs show up as a uniform score drop that looks like a model being bad, which
is the most expensive kind of benchmark bug to debug.
"""

from __future__ import annotations

from spark_eval.prompting import build_prompt, extract_code
from spark_eval.schema import Fixture, Task

SOLVE = "def solve(spark, df):\n    return df"


def test_fenced_python_block():
    assert extract_code(f"Here you go:\n\n```python\n{SOLVE}\n```\n") == SOLVE


def test_fence_without_language_tag():
    assert extract_code(f"```\n{SOLVE}\n```") == SOLVE


def test_unterminated_fence_still_extracts():
    """Hitting the token limit mid-block should not cost the model the task."""
    assert extract_code(f"```python\n{SOLVE}") == SOLVE


def test_picks_the_block_that_defines_solve():
    text = (
        "First, the schema:\n\n```python\nschema = 'a INT'\n```\n\n"
        f"And the answer:\n\n```python\n{SOLVE}\n```"
    )
    assert extract_code(text) == SOLVE


def test_reasoning_block_is_stripped():
    text = (
        "<think>\nMaybe ```python\ndef solve(spark, df): return None\n```\n"
        "no wait, that is wrong.\n</think>\n\n"
        f"```python\n{SOLVE}\n```"
    )
    assert extract_code(text) == SOLVE


def test_unterminated_reasoning_block_yields_nothing_runnable():
    """Ran out of budget while thinking -- must not score the scratchpad."""
    code = extract_code("<think>\nI should probably write\n```python\nx = 1\n```")
    assert "def solve" not in code


def test_bare_code_without_fence():
    text = f"Sure, this works.\n{SOLVE}"
    assert extract_code(text) == SOLVE


def test_empty_response():
    assert extract_code("") == ""
    assert extract_code("   \n  ") == ""


def test_prose_only_response_has_no_solve():
    assert "def solve" not in extract_code("I cannot help with that request.")


def test_prompt_names_every_fixture_and_the_signature():
    task = Task(
        id="t",
        category="joins",
        difficulty="easy",
        prompt="Join orders to customers.",
        fixtures=[
            Fixture(name="orders", schema="id INT", rows=[[1]]),
            Fixture(name="customers", schema="id INT", rows=[[1]]),
        ],
        solution=SOLVE,
    )
    prompt = build_prompt(task)
    assert "def solve(spark, orders, customers)" in prompt
    assert "orders: id INT" in prompt
    assert "customers: id INT" in prompt
