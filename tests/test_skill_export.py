"""Tests for jaxlint.skills.export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jaxlint.cli.main import app as main_app
from jaxlint.skills.export import (
    ExportError,
    ExportResult,
    SkillInfo,
    export_skill,
    get_skills_root,
    list_skills,
    register_mcp,
    resolve_target,
)


# ---------------------------------------------------------------------------
# get_skills_root
# ---------------------------------------------------------------------------

def test_get_skills_root_returns_existing_path():
    root = get_skills_root()
    assert root.exists(), f"skills root not found: {root}"
    assert root.is_dir()


def test_get_skills_root_contains_skill_dirs():
    root = get_skills_root()
    skill_dirs = [d.name for d in root.iterdir() if d.is_dir()]
    assert "using-jax" in skill_dirs


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------

def test_list_skills_returns_all_six():
    skills = list_skills()
    names = [s.name for s in skills]
    assert len(skills) >= 6
    assert "using-jax" in names
    assert "using-equinox" in names
    assert "using-orbax" in names
    assert "using-jaxlint" in names
    assert "developing-composable-jax" in names
    assert "exporting-jax" in names


def test_list_skills_have_descriptions():
    for skill in list_skills():
        assert skill.description, f"skill {skill.name!r} has empty description"


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------

def test_resolve_target_claude_user():
    target = resolve_target("using-jax", tool="claude", level="user")
    assert target == Path.home() / ".claude" / "skills" / "using-jax" / "SKILL.md"


def test_resolve_target_claude_workspace():
    target = resolve_target("using-jax", tool="claude", level="workspace")
    assert target == Path(".claude") / "skills" / "using-jax" / "SKILL.md"


def test_resolve_target_gemini_user():
    target = resolve_target("using-jax", tool="gemini", level="user")
    assert target == Path.home() / ".gemini" / "skills" / "using-jax" / "SKILL.md"


def test_resolve_target_gemini_workspace():
    target = resolve_target("using-jax", tool="gemini", level="workspace")
    assert target == Path(".gemini") / "skills" / "using-jax" / "SKILL.md"


def test_resolve_target_unknown_tool_raises():
    with pytest.raises(ExportError, match="Unknown tool"):
        resolve_target("using-jax", tool="cursor", level="user")


def test_resolve_target_unknown_level_raises():
    with pytest.raises(ExportError, match="Unknown level"):
        resolve_target("using-jax", tool="claude", level="enterprise")


def test_resolve_target_unknown_skill_raises():
    with pytest.raises(ExportError, match="Unknown skill"):
        resolve_target("not-a-skill", tool="claude", level="user")


# ---------------------------------------------------------------------------
# export_skill
# ---------------------------------------------------------------------------

def test_export_skill_dry_run_writes_nothing(tmp_path):
    target = tmp_path / "SKILL.md"
    result = export_skill("using-jax", target=target, dry_run=True)
    assert not target.exists()
    assert result.dry_run is True
    assert result.target == target


def test_export_skill_writes_file(tmp_path):
    target = tmp_path / "skills" / "using-jax" / "SKILL.md"
    result = export_skill("using-jax", target=target, dry_run=False)
    assert target.exists()
    assert result.dry_run is False
    content = target.read_text()
    assert "using-jax" in content


def test_export_skill_inserts_version_stamp(tmp_path):
    target = tmp_path / "SKILL.md"
    export_skill("using-jax", target=target, dry_run=False)
    content = target.read_text()
    assert "jaxlint v" in content
    assert "exported" in content


def test_export_skill_stamp_is_after_frontmatter(tmp_path):
    target = tmp_path / "SKILL.md"
    export_skill("using-jax", target=target, dry_run=False)
    content = target.read_text()
    # Stamp must come after the closing ---
    fm_end = content.index("---\n", content.index("---\n") + 3) + 4
    stamp_pos = content.index("<!-- jaxlint v")
    assert stamp_pos >= fm_end


def test_export_skill_creates_parent_dirs(tmp_path):
    target = tmp_path / "deep" / "nested" / "SKILL.md"
    export_skill("using-jax", target=target, dry_run=False)
    assert target.exists()


def test_export_skill_unknown_skill_raises(tmp_path):
    with pytest.raises(ExportError, match="Unknown skill"):
        export_skill("not-a-skill", target=tmp_path / "SKILL.md", dry_run=False)


# ---------------------------------------------------------------------------
# register_mcp
# ---------------------------------------------------------------------------

def test_register_mcp_dry_run_writes_nothing(tmp_path):
    cfg = tmp_path / ".claude.json"
    register_mcp(tool="claude", level="user", dry_run=True, _config_override=cfg)
    assert not cfg.exists()


def test_register_mcp_creates_claude_json(tmp_path):
    cfg = tmp_path / ".claude.json"
    register_mcp(tool="claude", level="user", dry_run=False, _config_override=cfg)
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "mcpServers" in data
    assert "jaxlint" in data["mcpServers"]
    assert "command" in data["mcpServers"]["jaxlint"]


def test_register_mcp_merges_existing_config(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"mcpServers": {"other-tool": {"command": "other"}}}))
    register_mcp(tool="claude", level="user", dry_run=False, _config_override=cfg)
    data = json.loads(cfg.read_text())
    assert "other-tool" in data["mcpServers"]
    assert "jaxlint" in data["mcpServers"]


def test_register_mcp_gemini_writes_settings(tmp_path):
    cfg = tmp_path / "settings.json"
    register_mcp(tool="gemini", level="user", dry_run=False, _config_override=cfg)
    data = json.loads(cfg.read_text())
    assert "mcpServers" in data
    assert "jaxlint" in data["mcpServers"]


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

runner = CliRunner()


def test_skill_list_command():
    result = runner.invoke(main_app, ["skill", "list"])
    assert result.exit_code == 0
    assert "using-jax" in result.output
    assert "using-equinox" in result.output


def test_skill_list_shows_descriptions():
    result = runner.invoke(main_app, ["skill", "list"])
    assert result.exit_code == 0
    # Each skill should have a non-empty description shown
    assert "Use when" in result.output


def test_skill_export_dry_run():
    result = runner.invoke(
        main_app,
        ["skill", "export", "using-jax", "--tool", "claude", "--level", "user", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Dry run" in result.output or "dry-run" in result.output.lower()


def test_skill_export_all_dry_run():
    result = runner.invoke(
        main_app,
        ["skill", "export", "--all", "--tool", "claude", "--level", "user", "--dry-run"],
    )
    assert result.exit_code == 0
    # Should mention all 6 skills
    assert "using-jax" in result.output
    assert "exporting-jax" in result.output


def test_skill_export_unknown_skill_exits_nonzero():
    result = runner.invoke(
        main_app,
        ["skill", "export", "nonexistent-skill", "--tool", "claude", "--level", "user"],
    )
    assert result.exit_code == 1


def test_skill_export_writes_file(tmp_path):
    target = tmp_path / "using-jax" / "SKILL.md"
    result = runner.invoke(
        main_app,
        [
            "skill", "export", "using-jax",
            "--tool", "claude", "--level", "user",
            "--target-dir", str(tmp_path),
            "--no-mcp",
        ],
    )
    assert result.exit_code == 0
    assert target.exists()
