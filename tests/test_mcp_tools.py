"""MCP tool helpers (requires ``jaxlint[mcp]``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from jaxlint import __version__
from jaxlint.core.rules import describe_rules
from jaxlint.mcp.server import jaxlint_check_impl, jaxlint_rules_text, jaxlint_version_text

pytestmark = pytest.mark.mcp


def test_version_string_nonempty() -> None:
    assert jaxlint_version_text().strip() == __version__.strip()
    assert __version__


def test_rules_include_jl001() -> None:
    text = jaxlint_rules_text()
    assert "JL001" in text
    assert text == describe_rules()


def test_check_path_jd001(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "jd001_missing_docstring.py"
    dst = tmp_path / "jd001_missing_docstring.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    out = jaxlint_check_impl([str(dst)], None, "snippet.py")
    assert not out.startswith("error:")
    payload = json.loads(out)
    assert any(item.get("rule_id") == "JD001" for item in payload)


def test_check_source_only_jd001(fixtures_dir: Path) -> None:
    text = (fixtures_dir / "jd001_missing_docstring.py").read_text(encoding="utf-8")
    out = jaxlint_check_impl(None, text, "snippet.py")
    assert not out.startswith("error:")
    payload = json.loads(out)
    assert any(item.get("rule_id") == "JD001" for item in payload)
