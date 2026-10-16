"""MCP server (stdio FastMCP) for jaxlint — optional ``jaxlint[mcp]`` extra."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    # mcp 2.x renamed FastMCP to MCPServer; the surface used here (constructor name,
    # @tool(), run() defaulting to stdio) is unchanged. 2.x is what resolves on
    # Python >= 3.13 alongside the `plugin` extra (cisternal -> fastmcp -> mcp 2).
    from mcp.server.mcpserver import MCPServer as _Server
except ModuleNotFoundError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from jaxlint import __version__
from jaxlint.core.config import load_config
from jaxlint.core.diagnostic import Diagnostic
from jaxlint.core.rules import describe_rules
from jaxlint.core.runner import diagnostics_for_python_source, format_json, run_checks

mcp = _Server("jaxlint")

MCP_SCHEMA_VERSION = "jaxlint.mcp.v1"

# Single source of truth for ``jaxlint_capabilities`` (tool names, descriptions, parameter intent).
MCP_TOOLS_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "jaxlint_check",
        "description": (
            "Run jaxlint on filesystem paths or in-memory Python source; returns diagnostic JSON "
            "(legacy array) or a versioned structured envelope when structured=true."
        ),
        "parameters": [
            {
                "name": "paths",
                "type": "list[str] | None",
                "intent": "Optional list of files or directories to lint (path mode).",
            },
            {
                "name": "source",
                "type": "str | None",
                "intent": "Optional Python source string (source mode); mutually exclusive with paths.",
            },
            {
                "name": "virtual_path",
                "type": "str",
                "intent": 'Anchor path for config discovery when using source mode (default "snippet.py").',
            },
            {
                "name": "structured",
                "type": "bool",
                "intent": "If true, return a jaxlint.mcp.v1 JSON object with diagnostics, meta, and optional error.",
            },
            {
                "name": "jobs",
                "type": "int | None",
                "intent": "Parallel worker count for path mode only; forwarded to run_checks (omit for default).",
            },
            {
                "name": "cache_dir",
                "type": "str | None",
                "intent": "Optional disk cache directory for path mode only; forwarded to run_checks.",
            },
        ],
    },
    {
        "name": "jaxlint_capabilities",
        "description": "Return jaxlint.mcp.v1 capability document (schema, package version, tools metadata).",
        "parameters": [],
    },
    {
        "name": "jaxlint_rules",
        "description": "List jaxlint rule IDs with severity and short description (tab-separated).",
        "parameters": [],
    },
    {
        "name": "jaxlint_version",
        "description": "Return the jaxlint package version string.",
        "parameters": [],
    },
]


def jaxlint_version_text() -> str:
    return __version__


def jaxlint_rules_text() -> str:
    return describe_rules()


def jaxlint_capabilities_text() -> str:
    """JSON string: schema version, package version, and registered MCP tools metadata."""
    payload = {
        "schema_version": MCP_SCHEMA_VERSION,
        "package_version": __version__,
        "tools": MCP_TOOLS_REGISTRY,
    }
    return json.dumps(payload, indent=2)


def _diagnostics_as_list(diags: list[Diagnostic]) -> list[dict[str, Any]]:
    return json.loads(format_json(diags))


def _structured_response(
    *,
    success: bool,
    diagnostics: list[dict[str, Any]],
    error: dict[str, str] | None,
    elapsed_ms: float,
    paths_resolved: int,
    jobs: int | None,
    cache_dir: str | None,
) -> str:
    payload = {
        "schema_version": MCP_SCHEMA_VERSION,
        "success": success,
        "diagnostics": diagnostics,
        "error": error,
        "meta": {
            "elapsed_ms": elapsed_ms,
            "paths_resolved": paths_resolved,
            "jobs": jobs,
            "cache_dir": cache_dir,
        },
    }
    return json.dumps(payload, indent=2)


def jaxlint_check_impl(
    paths: list[str] | None,
    source: str | None,
    virtual_path: str,
    *,
    structured: bool = False,
    jobs: int | None = None,
    cache_dir: str | None = None,
) -> str:
    """Run checks on ``paths`` or in-memory ``source``; returns JSON text or an error message."""
    t0 = time.perf_counter()
    requested_jobs = jobs
    requested_cache_dir = cache_dir

    def elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    has_paths = bool(paths)
    has_source = source is not None

    if has_paths and has_source:
        msg = "provide either `paths` or `source`, not both"
        if not structured:
            return f"error: {msg}"
        return _structured_response(
            success=False,
            diagnostics=[],
            error={"code": "validation_error", "message": msg},
            elapsed_ms=elapsed_ms(),
            paths_resolved=0,
            jobs=requested_jobs,
            cache_dir=requested_cache_dir,
        )
    if not has_paths and not has_source:
        msg = (
            "provide a non-empty `paths` list or a `source` string "
            "(optional `virtual_path` anchors config discovery)"
        )
        if not structured:
            return f"error: {msg}"
        return _structured_response(
            success=False,
            diagnostics=[],
            error={"code": "validation_error", "message": msg},
            elapsed_ms=elapsed_ms(),
            paths_resolved=0,
            jobs=requested_jobs,
            cache_dir=requested_cache_dir,
        )

    if has_source:
        if jobs is not None or cache_dir is not None:
            msg = "jobs and cache_dir apply only to path-based checks"
            if not structured:
                return f"error: {msg}"
            return _structured_response(
                success=False,
                diagnostics=[],
                error={"code": "unsupported_option", "message": msg},
                elapsed_ms=elapsed_ms(),
                paths_resolved=0,
                jobs=requested_jobs,
                cache_dir=requested_cache_dir,
            )
        vp = Path(virtual_path)
        cfg = load_config(vp.parent)
        diags = diagnostics_for_python_source(vp, source, cfg=cfg)
        if not structured:
            return format_json(diags)
        return _structured_response(
            success=True,
            diagnostics=_diagnostics_as_list(diags),
            error=None,
            elapsed_ms=elapsed_ms(),
            paths_resolved=0,
            jobs=requested_jobs,
            cache_dir=requested_cache_dir,
        )

    assert paths is not None
    resolved_paths = [Path(p).expanduser().resolve() for p in paths]
    anchor = resolved_paths[0].parent if resolved_paths else Path.cwd()
    cfg = load_config(anchor)
    cache_path = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None

    try:
        diags = run_checks(resolved_paths, cfg=cfg, jobs=jobs, cache_dir=cache_path)
    except ValueError as e:
        if not structured:
            return f"error: {e}"
        return _structured_response(
            success=False,
            diagnostics=[],
            error={"code": "invalid_argument", "message": str(e)},
            elapsed_ms=elapsed_ms(),
            paths_resolved=len(paths),
            jobs=requested_jobs,
            cache_dir=requested_cache_dir,
        )

    if not structured:
        return format_json(diags)
    return _structured_response(
        success=True,
        diagnostics=_diagnostics_as_list(diags),
        error=None,
        elapsed_ms=elapsed_ms(),
        paths_resolved=len(paths),
        jobs=requested_jobs,
        cache_dir=requested_cache_dir,
    )


@mcp.tool()
def jaxlint_check(
    paths: list[str] | None = None,
    source: str | None = None,
    virtual_path: str = "snippet.py",
    structured: bool = False,
    jobs: int | None = None,
    cache_dir: str | None = None,
) -> str:
    """Run jaxlint on file paths or in-memory Python source; returns JSON diagnostics."""
    return jaxlint_check_impl(
        paths,
        source,
        virtual_path,
        structured=structured,
        jobs=jobs,
        cache_dir=cache_dir,
    )


@mcp.tool()
def jaxlint_capabilities() -> str:
    """Describe jaxlint MCP schema version, package version, and exposed tools."""
    return jaxlint_capabilities_text()


@mcp.tool()
def jaxlint_rules() -> str:
    """List jaxlint rule IDs with severity and short description (tab-separated)."""
    return jaxlint_rules_text()


@mcp.tool()
def jaxlint_version() -> str:
    """Return the jaxlint package version string."""
    return jaxlint_version_text()


def main() -> None:
    mcp.run()
