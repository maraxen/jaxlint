"""Cover ``extract_math_only`` helper."""

from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.runner import extract_math_only


@pytest.mark.unit
def test_extract_math_only_respects_ignore() -> None:
    cfg = LintConfig(select=frozenset(["JL"]))
    txt = "$unclosed_without_close"
    out = extract_math_only(txt, Path("x.py"), cfg)
    assert out == []


@pytest.mark.unit
def test_extract_math_only_runs_when_math_selected() -> None:
    cfg = LintConfig(select=frozenset(["JM"]))
    txt = "$unclosed_without_close"
    out = extract_math_only(txt, Path("x.py"), cfg)
    assert out and out[0].rule_id == "JM001"
