"""Typer sub-app for `jaxlint skill` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from jaxlint.skills.export import (
    ExportError,
    export_skill,
    list_skills,
    register_mcp,
    resolve_target,
)

skill_app = typer.Typer(
    name="skill",
    help="Manage and export jaxlint skills to Claude/Gemini tool directories.",
    no_args_is_help=True,
)


@skill_app.command("list")
def list_cmd() -> None:
    """List all available skills with their descriptions."""
    skills = list_skills()
    if not skills:
        typer.echo("No skills found.", err=True)
        raise typer.Exit(1)
    max_name = max(len(s.name) for s in skills)
    for skill in skills:
        typer.echo(f"  {skill.name:<{max_name}}  {skill.description}")


@skill_app.command("export")
def export_cmd(
    skill_name: str | None = typer.Argument(
        default=None,
        help="Skill name to export (e.g. 'using-jax'). Omit with --all to export everything.",
    ),
    all_skills: bool = typer.Option(False, "--all", help="Export all available skills."),
    tool: str = typer.Option("claude", "--tool", "-t", help="Target tool: claude or gemini"),
    level: str = typer.Option(
        "user", "--level", "-l", help="Install level: user or workspace"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would happen without writing any files."
    ),
    no_mcp: bool = typer.Option(
        False, "--no-mcp", help="Skip MCP server registration."
    ),
    target_dir: Path | None = typer.Option(
        None,
        "--target-dir",
        hidden=True,
        help="Override base directory for skill targets (testing only).",
    ),
) -> None:
    """Export one or all skills to the tool's skills directory.

    Installs SKILL.md files and (unless --no-mcp) registers jaxlint-mcp
    in the tool's MCP config.
    """
    if not skill_name and not all_skills:
        typer.echo(
            "Specify a skill name or pass --all. "
            "Use `jaxlint skill list` to see available skills.",
            err=True,
        )
        raise typer.Exit(2)

    if skill_name and all_skills:
        typer.echo("Cannot specify both a skill name and --all.", err=True)
        raise typer.Exit(2)

    skills_to_export = list_skills()
    if skill_name:
        skills_to_export = [s for s in skills_to_export if s.name == skill_name]
        if not skills_to_export:
            typer.echo(
                f"Unknown skill: {skill_name!r}. "
                f"Run `jaxlint skill list` to see available skills.",
                err=True,
            )
            raise typer.Exit(1)

    errors: list[str] = []
    for skill in skills_to_export:
        try:
            if target_dir is not None:
                target = target_dir / skill.name / "SKILL.md"
            else:
                target = resolve_target(skill.name, tool=tool, level=level)
            result = export_skill(skill.name, target=target, dry_run=dry_run)
            prefix = "Dry run — would export" if dry_run else "Exported"
            typer.echo(f"{prefix} {skill.name!r:40} → {result.target}")
        except ExportError as e:
            typer.echo(f"Error exporting {skill.name!r}: {e}", err=True)
            errors.append(skill.name)

    if not no_mcp and not errors:
        try:
            cfg = register_mcp(tool=tool, level=level, dry_run=dry_run)
            prefix = "Dry run — would register" if dry_run else "Registered"
            typer.echo(f"{prefix} jaxlint-mcp in {cfg}")
        except ExportError as e:
            typer.echo(f"Warning: MCP registration failed: {e}", err=True)

    if errors:
        raise typer.Exit(1)
