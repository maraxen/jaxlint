"""`jaxlint plugin install|update|export|info` -- cisternal's shared plugin sub-app.

Same surface as `bth plugin`, `mrx plugin` and `myxcel plugin`. The command forwards its raw
argv (help flag included) to cisternal's ``plugin_app``, so cisternal owns the help text, the
error wording and the exit codes. The bundle resolves from the source checkout's
``.praxia/manifest.toml`` when run from one, else from the packaged
``jaxlint/agent_plugin.json`` snapshot (regenerate with
``cisternal assets snapshot --out src/jaxlint/agent_plugin.json``).

cisternal is the optional ``plugin`` extra and is imported only when this command runs, so
``jaxlint check`` and ``jaxlint.core`` never load it.
"""

from __future__ import annotations

import typer

PLUGIN_NAME = "jaxlint"

_MISSING_EXTRA = (
    "`jaxlint plugin` needs cisternal (Python >= 3.13), which is not installed.\n"
    "Install the extra, e.g.:  uv tool install --force 'jaxlint[plugin]'"
)


def plugin_cmd(ctx: typer.Context) -> None:
    """Install, update, or export the jaxlint agent plugin (`jaxlint plugin --help`)."""
    try:
        from cisternal.plugin import PluginSpec, plugin_app
    except ImportError:
        typer.echo(_MISSING_EXTRA, err=True)
        raise typer.Exit(code=1) from None
    spec = PluginSpec(name=PLUGIN_NAME, package="jaxlint", cli="jaxlint")
    plugin_app(spec)(list(ctx.args))
