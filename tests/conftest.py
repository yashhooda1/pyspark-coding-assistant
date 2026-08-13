from __future__ import annotations

from pathlib import Path

import pytest

from spark_eval.runner import get_spark

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


@pytest.fixture(scope="session")
def spark():
    """One session for the whole test run. JVM startup dominates otherwise."""
    session = get_spark("spark-eval-tests")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
