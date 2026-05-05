"""Collect Python paths and run enabled checks."""

from __future__ import annotations

import json
from pathlib import Path

from jaxlint.core.config import LintConfig, load_config, rule_enabled
from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.doc.mathjax import mathjax_diagnostics
from jaxlint.core.doc.sections import doc_diagnostics_for_file
from jaxlint.core.sensors import perf_diagnostics_for_file

SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
)


def resolve_paths(paths: list[Path]) -> list[Path]:
    py_files: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp.is_file() and rp.suffix == ".py":
            py_files.append(rp)
        elif rp.is_dir():
            py_files.extend(sorted(rp.rglob("*.py")))
    return sorted(set(py_files))


def format_text(diagnostics: list[Diagnostic]) -> str:
    lines = []
    for d in sorted(diagnostics, key=lambda x: x.sort_key()):
        loc = f"{d.path}:{d.line}:{d.col}"
        lines.append(f"{loc} [{d.severity.value}] {d.rule_id} — {d.message}")
    return "\n".join(lines)


def format_json(diagnostics: list[Diagnostic]) -> str:
    payload = [
        {
            "rule_id": d.rule_id,
            "severity": d.severity.value,
            "message": d.message,
            "path": str(d.path),
            "line": d.line,
            "col": d.col,
            "end_line": d.end_line,
            "end_col": d.end_col,
        }
        for d in sorted(diagnostics, key=lambda x: x.sort_key())
    ]
    return json.dumps(payload, indent=2)


def format_github_annotations_json(diagnostics: list[Diagnostic]) -> str:
    """JSON shaped for GitHub Checks `output.annotations` (REST API field names).

    Omits ``start_column`` / ``end_column`` when ``start_line != end_line`` (GitHub
    requires columns only for single-line annotations). Uses 1-based ``start_column``
    / ``end_column``; ``col`` / ``end_col`` of ``0`` are treated as start of line (column 1).
    """

    def annotation_level(sev: Severity) -> str:
        if sev == Severity.ERROR:
            return "failure"
        if sev == Severity.WARNING:
            return "warning"
        return "notice"

    def one_based_col(c: int) -> int:
        return 1 if c <= 0 else c

    annotations: list[dict] = []
    for d in sorted(diagnostics, key=lambda x: x.sort_key()):
        end_line = d.end_line if d.end_line is not None else d.line
        row: dict = {
            "path": str(d.path),
            "start_line": d.line,
            "end_line": end_line,
            "annotation_level": annotation_level(d.severity),
            "message": d.message,
            "title": d.rule_id,
        }
        if end_line == d.line:
            row["start_column"] = one_based_col(d.col)
            if d.end_col is not None:
                row["end_column"] = one_based_col(d.end_col)
        annotations.append(row)
    return json.dumps({"annotations": annotations}, indent=2)


def format_compact(diagnostics: list[Diagnostic]) -> str:
    """One line per diagnostic: ``path:line:col: severity: message (rule_id)``."""
    lines = []
    for d in sorted(diagnostics, key=lambda x: x.sort_key()):
        loc = f"{d.path}:{d.line}:{d.col}"
        lines.append(f"{loc}: {d.severity.value}: {d.message} ({d.rule_id})")
    return "\n".join(lines)


def format_sarif(diagnostics: list[Diagnostic], *, tool_version: str) -> str:
    """Minimal SARIF 2.1.0 log (best-effort for generic viewers; not GHAS-specific)."""

    def sarif_level(sev: Severity) -> str:
        if sev == Severity.ERROR:
            return "error"
        if sev == Severity.WARNING:
            return "warning"
        return "note"

    def one_based_col(c: int) -> int:
        return 1 if c <= 0 else c

    results: list[dict] = []
    for d in sorted(diagnostics, key=lambda x: x.sort_key()):
        uri = d.path.resolve().as_uri()
        end_line = d.end_line if d.end_line is not None else d.line
        region: dict[str, int] = {"startLine": d.line, "endLine": end_line}
        # Match GitHub annotation rules: columns only when the span is a single line.
        if end_line == d.line:
            region["startColumn"] = one_based_col(d.col)
            if d.end_col is not None:
                region["endColumn"] = one_based_col(d.end_col)

        results.append(
            {
                "ruleId": d.rule_id,
                "level": sarif_level(d.severity),
                "message": {"text": d.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": uri},
                            "region": region,
                        }
                    }
                ],
            }
        )

    payload = {
        "$schema": SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "jaxlint",
                        "version": tool_version,
                        "informationUri": "https://github.com/jax-ml/jaxlint",
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(payload, indent=2)


def format_diagnostics(diagnostics: list[Diagnostic], format_name: str) -> str:
    fmt = format_name.strip().lower()
    if fmt == "text":
        return format_text(diagnostics)
    if fmt == "compact":
        return format_compact(diagnostics)
    if fmt == "json":
        return format_json(diagnostics)
    if fmt in ("github", "github-annotations"):
        return format_github_annotations_json(diagnostics)
    if fmt in ("sarif", "sarif-2.1.0"):
        from jaxlint import __version__ as jaxlint_version

        return format_sarif(diagnostics, tool_version=jaxlint_version)
    msg = f"Unknown --format {format_name!r} (expected text, compact, json, github, or sarif)"
    raise ValueError(msg)


def run_checks(
    paths: list[Path],
    *,
    cfg: LintConfig | None = None,
    perf: bool = True,
    doc: bool = True,
    mathjx: bool = True,
    config_start: Path | None = None,
) -> list[Diagnostic]:
    cfg = cfg or load_config(config_start)
    diagnostics: list[Diagnostic] = []
    py_files = resolve_paths(paths)

    for py in py_files:
        if perf:
            for d in perf_diagnostics_for_file(py, cfg):
                if rule_enabled(d.rule_id, cfg, py):
                    diagnostics.append(d)
        if doc or mathjx:
            for d in doc_diagnostics_for_file(py, cfg, sections=doc, mathjax=mathjx):
                if rule_enabled(d.rule_id, cfg, py):
                    diagnostics.append(d)
    return diagnostics


def extract_math_only(doc: str, path: Path, cfg: LintConfig) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for d in mathjax_diagnostics(doc, path, line_base=1):
        if rule_enabled(d.rule_id, cfg, path):
            out.append(d)
    return out
