from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.runner import format_json, format_text, resolve_paths


@pytest.fixture
def sample_diag(tmp_path: Path) -> list[Diagnostic]:
    shared = tmp_path / "same.py"
    return [
        Diagnostic("JM003", Severity.WARNING, "later", shared, line=40, col=0),
        Diagnostic("JL001", Severity.ERROR, "col tie-break", shared, line=9, col=2),
        Diagnostic("JL001", Severity.ERROR, "col first", shared, line=9, col=0),
    ]


@pytest.mark.unit
def test_resolve_paths_dirs_and_files(tmp_path: Path) -> None:
    a = tmp_path / "x.py"
    b = tmp_path / "nested" / "y.py"
    b.parent.mkdir()
    ignore = tmp_path / "nested" / "skip.txt"
    ignore.write_text("x", encoding="utf-8")
    a.write_text("pass\n", encoding="utf-8")
    b.write_text("pass\n", encoding="utf-8")

    resolved = resolve_paths([tmp_path, a])
    assert set(resolved) == {a.resolve(), b.resolve()}


@pytest.mark.unit
def test_format_text_sorts_by_location(sample_diag: list[Diagnostic]) -> None:
    text = format_text(sample_diag)
    lines = text.splitlines()
    assert len(lines) == 3
    assert "[error] JL001" in lines[0] and ":9:0" in lines[0]
    assert "[error] JL001" in lines[1] and ":9:2" in lines[1]
    assert "JM003" in lines[-1] and ":40:0" in lines[-1]


@pytest.mark.unit
def test_format_json_roundtrip(sample_diag: list[Diagnostic]) -> None:
    raw = format_json(sample_diag)
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    assert {row["rule_id"] for row in parsed} == {"JL001", "JM003"}
    assert all(set(row.keys()) == {
        "rule_id",
        "severity",
        "message",
        "path",
        "line",
        "col",
        "end_line",
        "end_col",
    } for row in parsed)
    jl_rows = [r for r in parsed if r["rule_id"] == "JL001"]
    assert jl_rows[0]["line"] <= jl_rows[1]["line"]
