"""Typer command for `jaxlint export` — post-step hook for praxia plugin system."""

from __future__ import annotations

import typer


def export_cmd(
    surface: str = typer.Option(
        "claude_code",
        "--surface",
        help="Tool surface for export (e.g., claude_code, gemini, cursor)",
    ),
    level: str = typer.Option(
        "user",
        "--level",
        "-l",
        help="Install level: user or workspace",
    ),
) -> None:
    """Post-step hook for praxia plugin export.

    Confirms that jaxlint skills and snippets are ready for injection into
    the specified tool surface. This is a reporting hook — actual file writing
    is done by praxia's StandardWriter.

    Example:
        jaxlint export --surface claude_code --level user
    """
    typer.echo(f"jaxlint export confirmed: surface={surface}, level={level}")
    raise typer.Exit(0)
