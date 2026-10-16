"""`jaxlint plugin` forwards its raw argv to cisternal's plugin sub-app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jaxlint.cli.main import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_command_is_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "plugin" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["info"],
        ["--help"],
        ["install", "--scope", "user"],
        ["export", "claude", "--out", "/tmp/x"],
    ],
)
def test_plugin_forwards_argv_unparsed(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    cisternal_plugin = pytest.importorskip("cisternal.plugin")
    seen: dict[str, object] = {}

    def fake_plugin_app(spec):  # noqa: ANN001, ANN202
        seen["spec"] = spec

        def run(args: list[str]) -> None:
            seen["args"] = args

        return run

    monkeypatch.setattr(cisternal_plugin, "plugin_app", fake_plugin_app)
    result = runner.invoke(app, ["plugin", *argv])
    assert result.exit_code == 0, result.output
    assert seen["args"] == argv
    spec = seen["spec"]
    assert (spec.name, spec.package, spec.cli) == ("jaxlint", "jaxlint", "jaxlint")


def test_plugin_without_extra_explains_and_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import cisternal.plugin` raise ImportError.
    monkeypatch.setitem(sys.modules, "cisternal", None)
    monkeypatch.setitem(sys.modules, "cisternal.plugin", None)
    result = runner.invoke(app, ["plugin", "info"])
    assert result.exit_code == 1
    assert "jaxlint[plugin]" in result.output


def test_packaged_snapshot_lists_every_manifest_skill() -> None:
    """The wheel's agent_plugin.json must carry every skill the manifest declares.

    Regenerate with `cisternal assets snapshot --out src/jaxlint/agent_plugin.json`.
    """
    tomllib = pytest.importorskip("tomllib")
    manifest = tomllib.loads((REPO_ROOT / ".praxia/manifest.toml").read_text())
    declared = {s["name"] for s in manifest["plugin"]["skills"]}
    snapshot_text = (REPO_ROOT / "src/jaxlint/agent_plugin.json").read_text()
    snapshot = json.loads(snapshot_text)
    assert "bundle" in snapshot
    missing = {name for name in declared if f'"{name}"' not in snapshot_text}
    assert not missing, f"snapshot is stale; missing skills: {sorted(missing)}"
