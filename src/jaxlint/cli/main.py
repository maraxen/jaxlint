"""Typer CLI."""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import typer

from jaxlint import __version__
from jaxlint.core.config import load_config, starter_toml_snippet
from jaxlint.core.rules import describe_rules
from jaxlint.core.runner import format_diagnostics, resolve_paths, run_checks
from jaxlint.core.doc.rubric_impl import RubricScorer
from jaxlint.cli.skill_commands import skill_app
from jaxlint.cli.setup_command import setup_cmd
from jaxlint.cli.export_command import export_cmd

app = typer.Typer(help="JAX-oriented linting and doc quality", no_args_is_help=True)
app.add_typer(skill_app, name="skill")
app.command("setup")(setup_cmd)
app.command("export")(export_cmd)


def _config_anchor(config: Path | None) -> Path:
    """Directory used as ``run_checks(..., config_start=...)`` when config is supplied."""
    if config is None:
        return Path.cwd()
    resolved = Path(config).resolve()
    return resolved.parent if resolved.is_file() else resolved


@app.command("check")
def check_cmd(
    paths: list[Path] = typer.Argument(default=None, help="Files or directories (.py)"),
    no_doc: bool = typer.Option(False, "--no-doc", help="Skip docstring / MathJax checks"),
    no_perf: bool = typer.Option(False, "--no-perf", help="Skip performance AST sensors"),
    format_: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="text, compact, json, github (Checks JSON), sarif, rich",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Directory to search from, or path to jaxlint.toml / pyproject.toml",
    ),
    select: str | None = typer.Option(None, "--select", help="Comma-separated rule prefixes (override config)"),
    ignore: str | None = typer.Option(None, "--ignore", help="Comma-separated rule prefixes to ignore"),
    jobs: int | None = typer.Option(
        None,
        "--jobs",
        "-j",
        help="Parallel file workers (default: min(32, CPU count); use 1 for serial)",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help=(
            "Optional directory for on-disk diagnostic cache (overrides jaxlint.toml / pyproject "
            "cache-dir)"
        ),
    ),
) -> None:
    if not paths:
        paths = [Path(".")]
    cfg_source = Path(config).resolve() if config else None
    cfg = load_config(cfg_source)
    run_anchor = _config_anchor(config)
    if select:
        cfg = replace(cfg, select=frozenset(s.strip() for s in select.split(",") if s.strip()))
    if ignore:
        cfg = replace(cfg, ignore=frozenset(s.strip() for s in ignore.split(",") if s.strip()))
    effective_cache = cache_dir.resolve() if cache_dir is not None else cfg.cache_dir
    try:
        diags = run_checks(
            paths,
            cfg=cfg,
            perf=not no_perf,
            doc=not no_doc,
            mathjx=not no_doc,
            config_start=run_anchor,
            jobs=jobs,
            cache_dir=effective_cache,
        )
        typer.echo(format_diagnostics(diags, format_, cfg=cfg))
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    raise typer.Exit(1 if any(d.severity.value == "error" for d in diags) else 0)


@app.command("doc")
def doc_cmd(
    paths: list[Path] = typer.Argument(default=None),
    format_: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="text, compact, json, github (Checks JSON), sarif, rich",
    ),
    config: Path | None = typer.Option(None, "--config"),
    select: str | None = typer.Option(None, "--select"),
    ignore: str | None = typer.Option(None, "--ignore"),
) -> None:
    if not paths:
        paths = [Path(".")]
    cfg_source = Path(config).resolve() if config else None
    cfg = load_config(cfg_source)
    run_anchor = _config_anchor(config)
    if select:
        cfg = replace(cfg, select=frozenset(s.strip() for s in select.split(",") if s.strip()))
    if ignore:
        cfg = replace(cfg, ignore=frozenset(s.strip() for s in ignore.split(",") if s.strip()))
    diags = run_checks(paths, cfg=cfg, perf=False, doc=True, mathjx=True, config_start=run_anchor)
    try:
        typer.echo(format_diagnostics(diags, format_, cfg=cfg))
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e
    raise typer.Exit(1 if any(d.severity.value == "error" for d in diags) else 0)


@app.command("score")
def score_cmd(
    paths: list[Path] = typer.Argument(default=None, help="Python files to score"),
    jsonl: Path | None = typer.Option(None, "--jsonl", help="Evaluate a .jsonl of generated docstrings"),
    baseline: Path | None = typer.Option(None, "--baseline"),
    threshold: float | None = typer.Option(None, "--threshold"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    cfg_source = Path(config).resolve() if config else None
    cfg = load_config(cfg_source)
    th = threshold if threshold is not None else cfg.rubric_threshold
    scorer = RubricScorer()

    def load_jsonl_scores(path: Path) -> list[dict]:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                full_doc = _extract_doc_from_entry(entry.get("docstring"))
                sig = entry.get("signature") or ""
                s = scorer.score(full_doc, sig, style=cfg.docstring_style)
                rows.append(
                    {
                        "name": entry.get("name") or entry.get("signature", "unknown"),
                        "normalized_score": s.normalized_score,
                        "passes": s.normalized_score >= th,
                        "details": s.details,
                    }
                )
        return rows

    if jsonl:
        current = load_jsonl_scores(jsonl)
        scores = [r["normalized_score"] for r in current]
        mean = sum(scores) / len(scores) if scores else 0.0
        pr = sum(1 for r in current if r["normalized_score"] >= th) / len(current) if current else 0.0
        typer.echo(f"Mean score: {mean:.3f}  pass rate (>={th}): {pr:.1%}  N={len(current)}")
        if baseline and baseline.exists():
            base = load_jsonl_scores(baseline)
            bm = sum(r["normalized_score"] for r in base) / len(base) if base else 0
            bp = sum(1 for r in base if r["normalized_score"] >= th) / len(base) if base else 0
            typer.echo(f"Baseline mean: {bm:.3f}  pass rate: {bp:.1%}")
            typer.echo(f"Delta mean: {mean - bm:+.3f}")
        fails = sum(1 for r in current if r["normalized_score"] < th)
        raise typer.Exit(1 if fails else 0)

    if not paths:
        paths = [Path(".")]
    failed = False
    for py in resolve_paths(paths):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            typer.echo(f"{py}: syntax error")
            failed = True
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if not doc:
                    continue
                sig = ""
                if isinstance(node, ast.FunctionDef):
                    sig = _function_signature_stub(node)
                s = scorer.score(doc, sig, style=cfg.docstring_style)
                typer.echo(f"{py}::{node.name}\t{s.normalized_score:.3f}\t{s.grade}")
                if s.normalized_score < th:
                    failed = True

    raise typer.Exit(1 if failed else 0)


def _function_signature_stub(node: ast.FunctionDef) -> str:
    fields: dict = {
        "name": node.name,
        "args": node.args,
        "body": [ast.Pass()],
        "decorator_list": node.decorator_list,
        "returns": node.returns,
        "type_comment": getattr(node, "type_comment", None),
        "lineno": getattr(node, "lineno", 1),
        "col_offset": getattr(node, "col_offset", 0),
    }
    if sys.version_info >= (3, 12):
        fields["type_params"] = getattr(node, "type_params", [])
    stub = ast.FunctionDef(**fields)
    return ast.unparse(stub).replace("pass", "").strip().rstrip(":")


def _extract_doc_from_entry(docstring) -> str:  # type: ignore[no-untyped-def]
    if docstring is None:
        return ""
    if isinstance(docstring, dict):
        if "full_docstring" in docstring:
            full_doc = docstring.get("full_docstring", "") or ""
            if docstring.get("args"):
                full_doc += f"\n\nArgs:\n    {docstring['args']}"
            if docstring.get("returns"):
                full_doc += f"\n\nReturns:\n    {docstring['returns']}"
            return full_doc
        return f"{docstring.get('summary', '')}\n\n{docstring.get('description', '')}"
    return str(docstring)


@app.command("rules")
def rules_cmd() -> None:
    typer.echo(describe_rules())


@app.command("init")
def init_cmd() -> None:
    typer.echo(starter_toml_snippet())


@app.command("version")
def version_cmd() -> None:
    typer.echo(__version__)


@app.command("hlo")
def hlo_cmd() -> None:
    try:
        import jaxlint.hlo as hlo  # noqa: F401
    except ImportError as e:
        typer.echo(f"HLO extra not installed: {e}. Install with: uv sync --extra hlo", err=True)
        raise typer.Exit(2) from e
    typer.echo("Use jaxlint.hlo.GraphAuditSensor.audit_kernel(fn, *args) from Python.")


if __name__ == "__main__":
    app()
