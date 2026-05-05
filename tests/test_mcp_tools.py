"""MCP tool helpers (requires ``jaxlint[mcp]``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from jaxlint import __version__
from jaxlint.core.rules import describe_rules
from jaxlint.mcp.server import (
    jaxlint_capabilities_text,
    jaxlint_check_impl,
    jaxlint_rules_text,
    jaxlint_version_text,
)

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


def test_check_structured_path_jd001(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "jd001_missing_docstring.py"
    dst = tmp_path / "jd001_missing_docstring.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    out = jaxlint_check_impl([str(dst)], None, "snippet.py", structured=True)
    payload = json.loads(out)
    assert payload["schema_version"] == "jaxlint.mcp.v1"
    assert payload["success"] is True
    assert payload["error"] is None
    assert payload["meta"]["paths_resolved"] == 1
    assert any(item.get("rule_id") == "JD001" for item in payload["diagnostics"])


def test_check_structured_source_jd001(fixtures_dir: Path) -> None:
    text = (fixtures_dir / "jd001_missing_docstring.py").read_text(encoding="utf-8")
    out = jaxlint_check_impl(None, text, "snippet.py", structured=True)
    payload = json.loads(out)
    assert payload["schema_version"] == "jaxlint.mcp.v1"
    assert payload["success"] is True
    assert payload["meta"]["paths_resolved"] == 0
    assert any(item.get("rule_id") == "JD001" for item in payload["diagnostics"])


def test_source_mode_jobs_or_cache_dir_error(fixtures_dir: Path) -> None:
    text = (fixtures_dir / "jd001_missing_docstring.py").read_text(encoding="utf-8")
    raw = jaxlint_check_impl(None, text, "snippet.py", jobs=2)
    assert raw.startswith("error:")
    assert "path-based" in raw.lower()

    structured = jaxlint_check_impl(None, text, "snippet.py", structured=True, jobs=2)
    p = json.loads(structured)
    assert p["success"] is False
    assert p["diagnostics"] == []
    assert p["error"]["code"] == "unsupported_option"

    raw_cd = jaxlint_check_impl(None, text, "snippet.py", cache_dir="/tmp/jaxlint-mcp-cache-test")
    assert raw_cd.startswith("error:")
    assert "path-based" in raw_cd.lower()


def test_invalid_jobs_path_mode(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "jd001_missing_docstring.py"
    dst = tmp_path / "jd001_missing_docstring.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    raw = jaxlint_check_impl([str(dst)], None, "snippet.py", jobs=0)
    assert raw.startswith("error:")
    assert "jobs must be >= 1" in raw

    structured = jaxlint_check_impl([str(dst)], None, "snippet.py", structured=True, jobs=0)
    p = json.loads(structured)
    assert p["success"] is False
    assert p["error"]["code"] == "invalid_argument"
    assert "jobs must be >= 1" in p["error"]["message"]


def test_jaxlint_capabilities_json() -> None:
    raw = jaxlint_capabilities_text()
    payload = json.loads(raw)
    assert payload["schema_version"] == "jaxlint.mcp.v1"
    assert payload["package_version"] == __version__
    assert "tools" in payload
    names = {t["name"] for t in payload["tools"]}
    assert "jaxlint_check" in names
    assert "jaxlint_capabilities" in names


def test_check_path_with_cache_dir_smoke(tmp_path: Path, fixtures_dir: Path) -> None:
    src = fixtures_dir / "clean_minimal.py"
    dst = tmp_path / "clean_minimal.py"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    out = jaxlint_check_impl([str(dst)], None, "snippet.py", structured=True, cache_dir=str(cache))
    payload = json.loads(out)
    assert payload["success"] is True
    assert payload["meta"]["cache_dir"] == str(cache)
