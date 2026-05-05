"""Collect Python paths and run enabled checks."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from jaxlint.core.config import LintConfig, load_config, rule_enabled
from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.doc.mathjax import mathjax_diagnostics
from jaxlint.core.doc.sections import doc_diagnostics_for_file, doc_diagnostics_from_source
from jaxlint.core.parse_context import load_parsed_python_file
from jaxlint.core.sensors import perf_diagnostics_for_file, perf_diagnostics_from_tree

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


def format_rich(diagnostics: list[Diagnostic]) -> str:
    """Render diagnostics as a Rich table (plain ASCII when stdout is not a TTY)."""

    buf = StringIO()
    tty = sys.stdout.isatty()
    color = tty and not os.environ.get("NO_COLOR", "").strip()
    console = Console(
        file=buf,
        force_terminal=color,
        no_color=not color,
        highlight=False,
        width=120,
        soft_wrap=True,
        safe_box=not tty,
    )

    if not diagnostics:
        console.print(Text("No diagnostics.", style="dim" if color else ""))
        return buf.getvalue()

    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("Severity", style=None, no_wrap=True, min_width=8)
    table.add_column("Rule", no_wrap=True, min_width=6)
    table.add_column("Location", overflow="fold", min_width=24)
    table.add_column("Message", overflow="fold")

    def severity_style(sev: Severity) -> str:
        if sev == Severity.ERROR:
            return "bold red" if color else ""
        if sev == Severity.WARNING:
            return "yellow" if color else ""
        return "cyan" if color else ""

    for d in sorted(diagnostics, key=lambda x: x.sort_key()):
        loc = f"{d.path}:{d.line}:{d.col}"
        sev_text = Text(d.severity.value, style=severity_style(d.severity))
        table.add_row(sev_text, d.rule_id, loc, d.message)

    console.print(table)
    return buf.getvalue()


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
    if fmt == "rich":
        return format_rich(diagnostics)
    msg = f"Unknown --format {format_name!r} (expected text, compact, json, github, sarif, or rich)"
    raise ValueError(msg)


def _default_jobs() -> int:
    return min(32, os.process_cpu_count() or 1)


def _diagnostics_for_file(
    py: Path,
    cfg: LintConfig,
    *,
    perf: bool,
    doc: bool,
    mathjx: bool,
) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    want_doc = doc or mathjx

    if perf and want_doc:
        ctx = load_parsed_python_file(py)
        if ctx.tree is None:
            return []
        for d in perf_diagnostics_from_tree(py, cfg, ctx.tree):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(d)
        for d in doc_diagnostics_from_source(
            py, cfg, ctx.source, ctx.tree, sections=doc, mathjax=mathjx
        ):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(d)
        return out

    if perf:
        for d in perf_diagnostics_for_file(py, cfg):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(d)
    if want_doc:
        for d in doc_diagnostics_for_file(py, cfg, sections=doc, mathjax=mathjx):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(d)
    return out


def run_checks(
    paths: list[Path],
    *,
    cfg: LintConfig | None = None,
    perf: bool = True,
    doc: bool = True,
    mathjx: bool = True,
    config_start: Path | None = None,
    jobs: int | None = None,
) -> list[Diagnostic]:
    if jobs is not None and jobs < 1:
        msg = f"jobs must be >= 1, got {jobs}"
        raise ValueError(msg)

    cfg = cfg or load_config(config_start)
    py_files = resolve_paths(paths)
    effective = _default_jobs() if jobs is None else jobs

    def work(py: Path) -> list[Diagnostic]:
        return _diagnostics_for_file(py, cfg, perf=perf, doc=doc, mathjx=mathjx)

    batches: list[list[Diagnostic]] = []
    if not py_files:
        pass
    elif effective == 1:
        for py in py_files:
            batches.append(work(py))
    else:
        with ThreadPoolExecutor(max_workers=effective) as ex:
            batches = list(ex.map(work, py_files))

    diagnostics: list[Diagnostic] = [d for batch in batches for d in batch]
    diagnostics.sort(key=lambda d: d.sort_key())
    return diagnostics


def extract_math_only(doc: str, path: Path, cfg: LintConfig) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    for d in mathjax_diagnostics(doc, path, line_base=1):
        if rule_enabled(d.rule_id, cfg, path):
            out.append(d)
    return out
