"""Typer command for `jaxlint setup` — unified MCP + skill onboarding."""

from __future__ import annotations

import typer

from jaxlint.skills.export import (
    ExportError,
    export_skill,
    list_skills,
    register_mcp,
    register_mcp_claude_code,
    register_mcp_cursor,
    resolve_target,
)


def setup_cmd(
    tool: list[str] = typer.Option(
        ["claude-code", "gemini"],
        "--tool",
        "-t",
        help=(
            "Tool surfaces to set up (repeatable): claude-code, gemini, cursor. "
            "Default: claude-code and gemini."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview what would happen without writing files."
    ),
    no_mcp: bool = typer.Option(
        False, "--no-mcp", help="Skip MCP server registration; export skills only."
    ),
    no_skills: bool = typer.Option(
        False, "--no-skills", help="Skip skill export; register MCP only."
    ),
) -> None:
    """Set up jaxlint skills and MCP integration across tool surfaces.

    Exports skill SKILL.md files and registers jaxlint-mcp in each surface's
    config. Failures on individual surfaces are reported as warnings; other
    surfaces continue. Exit 0 if all succeed, exit 1 if any surface failed.
    """
    surfaces = list(dict.fromkeys(tool))
    known = {"claude-code", "gemini", "cursor"}
    errors: list[str] = []

    for surface in surfaces:
        if surface not in known:
            typer.echo(
                f"Error: Unknown surface {surface!r}. Choose from: {sorted(known)}",
                err=True,
            )
            errors.append(surface)

    if errors:
        raise typer.Exit(1)

    # Phase 1: Skill export (cursor has no skill format)
    if not no_skills:
        skills = list_skills()
        for skill in skills:
            for surface in surfaces:
                if surface == "cursor":
                    continue
                try:
                    tool_name = "claude" if surface == "claude-code" else surface
                    target = resolve_target(skill.name, tool=tool_name, level="user")
                    result = export_skill(skill.name, target=target, dry_run=dry_run)
                    prefix = "Dry run — would export" if dry_run else "Exported"
                    typer.echo(f"{prefix} {skill.name:<40} → {result.target}")
                except ExportError as e:
                    typer.echo(
                        f"Warning: Failed to export {skill.name!r} to {surface}: {e}",
                        err=True,
                    )
                    errors.append(f"{skill.name}@{surface}")

    # Phase 2: MCP registration
    if not no_mcp:
        for surface in surfaces:
            try:
                if surface == "claude-code":
                    cfg = register_mcp_claude_code(dry_run=dry_run)
                    prefix = "Dry run — would register" if dry_run else "Registered"
                    typer.echo(f"{prefix} jaxlint-mcp (claude-code) → {cfg}")
                elif surface == "gemini":
                    cfg = register_mcp(tool="gemini", level="user", dry_run=dry_run)
                    prefix = "Dry run — would register" if dry_run else "Registered"
                    typer.echo(f"{prefix} jaxlint-mcp (gemini) → {cfg}")
                elif surface == "cursor":
                    cfg = register_mcp_cursor(dry_run=dry_run)
                    prefix = "Dry run — would register" if dry_run else "Registered"
                    typer.echo(f"{prefix} jaxlint-mcp (cursor) → {cfg.resolve()}")
            except ExportError as e:
                typer.echo(f"Warning: MCP registration failed for {surface}: {e}", err=True)
                errors.append(f"mcp@{surface}")

    if errors:
        raise typer.Exit(1)
