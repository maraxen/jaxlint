from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.runner import (
    SARIF_SCHEMA_URI,
    format_compact,
    format_diagnostics,
    format_github_annotations_json,
    format_json,
    format_rich,
    format_sarif,
    format_text,
    resolve_paths,
)


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


@pytest.mark.unit
def test_format_github_annotations_json_shape(sample_diag: list[Diagnostic]) -> None:
    raw = format_github_annotations_json(sample_diag)
    payload = json.loads(raw)
    ann = payload["annotations"]
    assert len(ann) == 3
    levels = {a["annotation_level"] for a in ann}
    assert levels == {"failure", "warning"}
    err = next(a for a in ann if a["title"] == "JL001" and a["start_line"] == 9 and a["start_column"] == 1)
    assert err["annotation_level"] == "failure"


@pytest.mark.unit
def test_format_github_annotations_multiline_omits_columns(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    diags = [
        Diagnostic("JD001", Severity.WARNING, "multi", p, line=1, col=4, end_line=3, end_col=1),
    ]
    ann = json.loads(format_github_annotations_json(diags))["annotations"]
    assert len(ann) == 1
    assert "start_column" not in ann[0]
    assert "end_column" not in ann[0]


@pytest.mark.unit
def test_format_github_annotations_info_is_notice(tmp_path: Path) -> None:
    p = tmp_path / "i.py"
    diags = [Diagnostic("JM001", Severity.INFO, "note", p, line=2, col=0)]
    ann = json.loads(format_github_annotations_json(diags))["annotations"]
    assert ann[0]["annotation_level"] == "notice"


@pytest.mark.unit
def test_format_sarif_structure_and_order(sample_diag: list[Diagnostic]) -> None:
    raw = format_sarif(sample_diag, tool_version="9.9.9-test")
    payload = json.loads(raw)
    assert payload["$schema"] == SARIF_SCHEMA_URI
    assert payload["version"] == "2.1.0"
    run = payload["runs"][0]
    assert run["tool"]["driver"]["name"] == "jaxlint"
    assert run["tool"]["driver"]["version"] == "9.9.9-test"
    results = run["results"]
    assert len(results) == 3
    assert results[0]["ruleId"] == "JL001"
    assert results[0]["level"] == "error"
    assert results[0]["message"]["text"] == "col first"
    assert "uri" in results[0]["locations"][0]["physicalLocation"]["artifactLocation"]
    uri = results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert uri.startswith("file:")
    reg = results[0]["locations"][0]["physicalLocation"]["region"]
    assert reg["startLine"] == 9 and reg["endLine"] == 9
    assert reg["startColumn"] == 1


@pytest.mark.unit
def test_format_sarif_multiline_region_omits_columns(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    diags = [
        Diagnostic("JD001", Severity.WARNING, "multi", p, line=1, col=4, end_line=3, end_col=1),
    ]
    payload = json.loads(format_sarif(diags, tool_version="1"))
    reg = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert reg["startLine"] == 1 and reg["endLine"] == 3
    assert "startColumn" not in reg


@pytest.mark.unit
def test_format_diagnostics_sarif_aliases(sample_diag: list[Diagnostic]) -> None:
    assert format_diagnostics(sample_diag, "sarif") == format_diagnostics(sample_diag, "sarif-2.1.0")
    assert json.loads(format_diagnostics(sample_diag, "sarif"))["version"] == "2.1.0"


@pytest.mark.unit
def test_format_compact_sorts_like_text(sample_diag: list[Diagnostic]) -> None:
    compact = format_compact(sample_diag)
    lines = compact.splitlines()
    assert len(lines) == 3
    assert lines[0].endswith("(JL001)")
    assert ":9:0: error:" in lines[0]
    assert "col first" in lines[0]
    assert lines[-1].endswith("(JM003)")


@pytest.mark.unit
def test_format_rich_empty() -> None:
    out = format_rich([])
    assert "No diagnostics" in out


@pytest.mark.unit
def test_format_rich_includes_columns_and_rules(sample_diag: list[Diagnostic]) -> None:
    out = format_rich(sample_diag)
    assert "Severity" in out and "Rule" in out and "Location" in out and "Message" in out
    assert "JL001" in out and "JM003" in out
    assert "error" in out and "warning" in out


@pytest.mark.unit
def test_format_diagnostics_dispatch(sample_diag: list[Diagnostic]) -> None:
    assert format_diagnostics(sample_diag, "text") == format_text(sample_diag)
    assert format_diagnostics(sample_diag, "compact") == format_compact(sample_diag)
    assert format_diagnostics(sample_diag, "json") == format_json(sample_diag)
    assert format_diagnostics(sample_diag, "rich") == format_rich(sample_diag)
    assert format_diagnostics(sample_diag, "github") == format_github_annotations_json(sample_diag)
    assert format_diagnostics(sample_diag, "github-annotations") == format_github_annotations_json(sample_diag)
    g = json.loads(format_diagnostics(sample_diag, "github"))
    assert "annotations" in g


@pytest.mark.unit
def test_format_diagnostics_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown --format"):
        format_diagnostics([], "nope")
