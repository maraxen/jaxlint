"""Collect Python paths and run enabled checks."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import StringIO, UnsupportedOperation
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jaxlint.core.cache import content_digest, store_diagnostics, try_load_diagnostics
from jaxlint.core.config import LintConfig, effective_severity, load_config, rule_enabled
from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.doc.mathjax import mathjax_diagnostics
from jaxlint.core.doc.sections import doc_diagnostics_from_source
from jaxlint.core.parse_context import ParsedPythonFile, load_parsed_python_file, parse_python_source
from jaxlint.core.rules import RULES
from jaxlint.core.sensors import perf_diagnostics_from_tree
from jaxlint.core.suppression import filter_suppressed

SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
)

_RULE_CATALOGUE: dict[str, tuple[Severity, str]] = {
    rule_id: (sev, desc) for rule_id, sev, desc in RULES
}

_RICH_WIDTH_CAP = 200
_RICH_WIDTH_MIN = 40
_RICH_WIDTH_FALLBACK_NON_TTY = 120


def _rich_console_width(*, is_tty: bool) -> int:
    """Width for Rich output: valid ``COLUMNS`` env → terminal columns when TTY → fallback."""
    raw = os.environ.get("COLUMNS")
    if raw is not None:
        try:
            n = int(str(raw).strip())
            if n >= _RICH_WIDTH_MIN:
                return min(n, _RICH_WIDTH_CAP)
        except ValueError:
            pass
    if is_tty:
        try:
            fd = sys.stdout.fileno()
            cols = os.get_terminal_size(fd).columns
            return min(max(cols, _RICH_WIDTH_MIN), _RICH_WIDTH_CAP)
        except (OSError, ValueError, AttributeError, UnsupportedOperation):
            pass
    return _RICH_WIDTH_FALLBACK_NON_TTY


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


def format_sarif(
    diagnostics: list[Diagnostic],
    *,
    tool_version: str,
    cfg: LintConfig | None = None,
) -> str:
    """Minimal SARIF 2.1.0 log (best-effort for generic viewers; not GHAS-specific)."""

    def sarif_level(sev: Severity) -> str:
        if sev == Severity.ERROR:
            return "error"
        if sev == Severity.WARNING:
            return "warning"
        return "note"

    def one_based_col(c: int) -> int:
        return 1 if c <= 0 else c

    sorted_diags = sorted(diagnostics, key=lambda x: x.sort_key())
    ordered_rule_ids: list[str] = []
    seen: set[str] = set()
    for d in sorted_diags:
        if d.rule_id not in seen:
            seen.add(d.rule_id)
            ordered_rule_ids.append(d.rule_id)
    rule_id_to_index = {rid: i for i, rid in enumerate(ordered_rule_ids)}

    docs_base = (cfg.sarif_rule_docs_base.rstrip("/") if cfg and cfg.sarif_rule_docs_base else None)

    rules: list[dict] = []
    for rid in ordered_rule_ids:
        row: dict = {"id": rid}
        cat = _RULE_CATALOGUE.get(rid)
        if cat is not None:
            _sev, short = cat
            row["shortDescription"] = {"text": short}
            row["defaultConfiguration"] = {"level": sarif_level(_sev)}
        else:
            row["shortDescription"] = {"text": "Unknown or custom rule"}
        if docs_base is not None:
            row["helpUri"] = f"{docs_base}/{rid}"
        rules.append(row)

    results: list[dict] = []
    for d in sorted_diags:
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
                "ruleIndex": rule_id_to_index[d.rule_id],
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

    driver: dict = {
        "name": "jaxlint",
        "version": tool_version,
        "informationUri": "https://github.com/maraxen/jaxlint",
    }
    if rules:
        driver["rules"] = rules

    payload = {
        "$schema": SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": driver},
                "results": results,
            }
        ],
    }
    return json.dumps(payload, indent=2)


def _osc8_file_link(uri: str, label: str) -> str:
    """OSC 8 hyperlink (terminal escapes); ``label`` is visible text."""
    return f"\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\"


def format_rich(diagnostics: list[Diagnostic]) -> str:
    """Render diagnostics as a Rich panel + table (ASCII-safe when stdout is not a TTY)."""

    tty = sys.stdout.isatty()

    class _Out(StringIO):
        def isatty(self) -> bool:
            return tty

    buf = _Out()
    color = tty and not os.environ.get("NO_COLOR", "").strip()
    width = _rich_console_width(is_tty=tty)
    console = Console(
        file=buf,
        force_terminal=color,
        no_color=not color,
        highlight=False,
        width=width,
        soft_wrap=True,
        safe_box=not tty,
    )

    if not diagnostics:
        console.print(Text("No diagnostics.", style="dim" if color else ""))
        return buf.getvalue()

    counts = Counter(d.severity for d in diagnostics)
    summary = (
        f"error={counts[Severity.ERROR]} "
        f"warning={counts[Severity.WARNING]} "
        f"info={counts[Severity.INFO]}"
    )

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

    use_link = tty and color

    for d in sorted(diagnostics, key=lambda x: x.sort_key()):
        # Show a short label in hyperlinks to avoid breaking filenames when
        # Rich wraps long paths inside the OSC8 escape sequence. Keep the
        # full path for non-hyperlink output.
        display_label = f"{d.path.name}:{d.line}:{d.col}"
        loc_plain = f"{d.path}:{d.line}:{d.col}"
        if use_link:
            try:
                uri = d.path.resolve().as_uri()
                loc_cell: Text | str = _osc8_file_link(uri, display_label)
            except (OSError, ValueError):
                loc_cell = display_label
        else:
            loc_cell = loc_plain

        sev_text = Text(d.severity.value, style=severity_style(d.severity))
        table.add_row(sev_text, d.rule_id, loc_cell, d.message)

    console.print(Panel(table, title="jaxlint", subtitle=summary))
    return buf.getvalue()


def format_diagnostics(
    diagnostics: list[Diagnostic], format_name: str, cfg: LintConfig | None = None
) -> str:
    """Format diagnostics for CLI / library use.

    ``cfg`` is used when ``format_name`` is SARIF — pass the same `LintConfig`
    as ``run_checks`` (or from ``load_config``) so optional ``sarif-rule-docs-base``
    can populate ``helpUri`` on rule descriptors.
    """
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

        return format_sarif(diagnostics, tool_version=jaxlint_version, cfg=cfg)
    if fmt == "rich":
        return format_rich(diagnostics)
    msg = f"Unknown --format {format_name!r} (expected text, compact, json, github, sarif, or rich)"
    raise ValueError(msg)


def _usable_process_cpu_count() -> int:
    """Prefer ``os.process_cpu_count()`` (3.13+), else ``os.cpu_count()``."""
    fn = getattr(os, "process_cpu_count", None)
    if fn is not None:
        n = fn()
        if n is not None and n > 0:
            return n
    n = os.cpu_count()
    return n if n and n > 0 else 1


def _default_jobs() -> int:
    return min(32, _usable_process_cpu_count())


def _cached_diagnostics_for_file(
    py: Path,
    cfg: LintConfig,
    *,
    perf: bool,
    doc: bool,
    mathjx: bool,
    cache_dir: Path | None,
) -> list[Diagnostic]:
    if cache_dir is None:
        return _diagnostics_for_file(py, cfg, perf=perf, doc=doc, mathjx=mathjx)
    try:
        raw = py.read_bytes()
    except OSError:
        return _diagnostics_for_file(py, cfg, perf=perf, doc=doc, mathjx=mathjx)
    digest = content_digest(raw)
    hit = try_load_diagnostics(
        py,
        cfg=cfg,
        perf=perf,
        doc=doc,
        mathjx=mathjx,
        cache_dir=cache_dir,
        content_sha256=digest,
    )
    if hit is not None:
        return hit
    # Parse the same bytes that were hashed so the key always matches the diagnostics.
    ctx = parse_python_source(py, raw.decode("utf-8"))
    out = _diagnostics_from_parsed(py, cfg, ctx, perf=perf, doc=doc, mathjx=mathjx)
    store_diagnostics(
        py,
        cfg=cfg,
        perf=perf,
        doc=doc,
        mathjx=mathjx,
        cache_dir=cache_dir,
        content_sha256=digest,
        diagnostics=out,
    )
    return out


def _diagnostics_from_parsed(
    py: Path,
    cfg: LintConfig,
    ctx: ParsedPythonFile,
    *,
    perf: bool,
    doc: bool,
    mathjx: bool,
) -> list[Diagnostic]:
    """Shared per-file logic for a :class:`ParsedPythonFile` (disk or in-memory).

    This is the single funnel for CLI, LSP, MCP and the runner API: inline
    ``# jaxlint: ignore[...]`` comments are applied here, after rule filtering.
    """
    raw = _unsuppressed_diagnostics_from_parsed(py, cfg, ctx, perf=perf, doc=doc, mathjx=mathjx)
    return filter_suppressed(raw, ctx.source)


def _unsuppressed_diagnostics_from_parsed(
    py: Path,
    cfg: LintConfig,
    ctx: ParsedPythonFile,
    *,
    perf: bool,
    doc: bool,
    mathjx: bool,
) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    want_doc = doc or mathjx

    if perf and want_doc:
        if ctx.tree is None:
            return []
        for d in perf_diagnostics_from_tree(py, cfg, ctx.tree):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(effective_severity(d, cfg))
        for d in doc_diagnostics_from_source(
            py, cfg, ctx.source, ctx.tree, sections=doc, mathjax=mathjx
        ):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(effective_severity(d, cfg))
        return out

    if perf:
        if ctx.tree is None:
            return []
        for d in perf_diagnostics_from_tree(py, cfg, ctx.tree):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(effective_severity(d, cfg))
    if want_doc:
        if ctx.tree is None:
            return []
        for d in doc_diagnostics_from_source(
            py, cfg, ctx.source, ctx.tree, sections=doc, mathjax=mathjx
        ):
            if rule_enabled(d.rule_id, cfg, py):
                out.append(effective_severity(d, cfg))
    return out


def _diagnostics_for_file(
    py: Path,
    cfg: LintConfig,
    *,
    perf: bool,
    doc: bool,
    mathjx: bool,
) -> list[Diagnostic]:
    ctx = load_parsed_python_file(py)
    return _diagnostics_from_parsed(py, cfg, ctx, perf=perf, doc=doc, mathjx=mathjx)


def diagnostics_for_python_source(
    path: Path,
    source: str,
    *,
    cfg: LintConfig | None = None,
    perf: bool = True,
    doc: bool = True,
    mathjx: bool = True,
    config_start: Path | None = None,
) -> list[Diagnostic]:
    """Lint in-memory Python source for ``path`` (unsaved buffers, LSP, etc.).

    Semantics match :func:`_diagnostics_for_file` for the same ``path``, flags,
    and config, but ``source`` is used instead of reading the file from disk.
    Results are sorted by :meth:`Diagnostic.sort_key` (same contract as
    :func:`run_checks` for a single path).

    The language server does **not** use the on-disk diagnostic cache
    (``cache_dir``); callers should omit caching for editor buffers.
    """

    cfg = cfg if cfg is not None else load_config(config_start)
    ctx = parse_python_source(path, source)
    out = _diagnostics_from_parsed(path, cfg, ctx, perf=perf, doc=doc, mathjx=mathjx)
    out.sort(key=lambda d: d.sort_key())
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
    cache_dir: Path | None = None,
) -> list[Diagnostic]:
    if jobs is not None and jobs < 1:
        msg = f"jobs must be >= 1, got {jobs}"
        raise ValueError(msg)

    cfg = cfg or load_config(config_start)
    py_files = resolve_paths(paths)
    effective = _default_jobs() if jobs is None else jobs

    def work(py: Path) -> list[Diagnostic]:
        return _cached_diagnostics_for_file(
            py,
            cfg,
            perf=perf,
            doc=doc,
            mathjx=mathjx,
            cache_dir=cache_dir,
        )

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
            out.append(effective_severity(d, cfg))
    return out
