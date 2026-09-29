"""The Haiku tests use the real SDK's exception classes, so CI must install the ``llm`` group.

Those tests skip when ``anthropic`` is missing (a plain ``uv sync`` does not
install it). This file does not skip: in CI a missing SDK is a failure, so
the skip cannot hide every retry and leak test at once.
"""

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_ci_installs_the_llm_group():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv sync --locked --group llm" in workflow


def test_anthropic_is_importable_in_ci():
    if os.environ.get("CI"):
        assert importlib.util.find_spec("anthropic") is not None
