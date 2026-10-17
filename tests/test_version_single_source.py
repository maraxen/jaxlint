"""One version source: pyproject.toml. The manifest, the plugin bundle and the source fallback must match it."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _manifest() -> dict:
    return tomllib.loads((REPO_ROOT / ".praxia/manifest.toml").read_text())


def test_manifest_version_matches_pyproject() -> None:
    assert _manifest()["plugin"]["version"] == _pyproject()["project"]["version"]


def test_plugin_bundle_version_matches_pyproject() -> None:
    bundle = json.loads((REPO_ROOT / "src/jaxlint/agent_plugin.json").read_text())
    assert bundle["bundle"]["metadata"]["version"] == _pyproject()["project"]["version"]


def test_source_fallback_version_matches_pyproject() -> None:
    init_text = (REPO_ROOT / "src/jaxlint/__init__.py").read_text()
    match = re.search(r'__version__ = "([^"]+)"\s*$', init_text, flags=re.MULTILINE)
    assert match is not None, "no hard-coded fallback __version__ found"
    assert match.group(1) == _pyproject()["project"]["version"]


def test_plugin_mcp_command_matches_project_script() -> None:
    pyproject = _pyproject()
    command = _manifest()["plugin"]["mcp"]["command"]
    assert len(command) == 1
    assert command[0] in pyproject["project"]["scripts"]
