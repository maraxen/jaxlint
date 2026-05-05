"""MCP server (stdio FastMCP) for jaxlint — optional ``jaxlint[mcp]`` extra."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from jaxlint import __version__
from jaxlint.core.config import load_config
from jaxlint.core.rules import describe_rules
from jaxlint.core.runner import diagnostics_for_python_source, format_json, run_checks

mcp = FastMCP("jaxlint")


def jaxlint_version_text() -> str:
    return __version__


def jaxlint_rules_text() -> str:
    return describe_rules()


def jaxlint_check_impl(
    paths: list[str] | None,
    source: str | None,
    virtual_path: str,
) -> str:
    """Run checks on ``paths`` or in-memory ``source``; returns JSON text or an error message."""
    has_paths = bool(paths)
    has_source = source is not None

    if has_paths and has_source:
        return "error: provide either `paths` or `source`, not both"
    if not has_paths and not has_source:
        return (
            "error: provide a non-empty `paths` list or a `source` string "
            "(optional `virtual_path` anchors config discovery)"
        )

    if has_source:
        vp = Path(virtual_path)
        cfg = load_config(vp.parent)
        diags = diagnostics_for_python_source(vp, source, cfg=cfg)
        return format_json(diags)

    assert paths is not None
    resolved = [Path(p).expanduser().resolve() for p in paths]
    anchor = resolved[0].parent if resolved else Path.cwd()
    cfg = load_config(anchor)
    diags = run_checks(resolved, cfg=cfg)
    return format_json(diags)


@mcp.tool()
def jaxlint_check(
    paths: list[str] | None = None,
    source: str | None = None,
    virtual_path: str = "snippet.py",
) -> str:
    """Run jaxlint on file paths or in-memory Python source; returns JSON diagnostics."""
    return jaxlint_check_impl(paths, source, virtual_path)


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
