"""Tests for jaxlint.skills.export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jaxlint.cli.main import app as main_app
from jaxlint.skills.export import (
    ExportError,
    export_skill,
    get_skills_root,
    list_skills,
    register_mcp,
    register_mcp_claude_code,
    register_mcp_cursor,
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


# ---------------------------------------------------------------------------
# register_mcp_claude_code
# ---------------------------------------------------------------------------

def test_register_mcp_claude_code_subprocess_success(monkeypatch):
    """subprocess success: calls 'claude mcp add jaxlint <cmd>', returns claude path."""
    import subprocess

    mcp_calls = []

    def mock_run(args, **kwargs):
        mcp_calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)

    result = register_mcp_claude_code(dry_run=False)

    assert len(mcp_calls) == 1
    assert mcp_calls[0][0:4] == ["claude", "mcp", "add", "jaxlint"]
    assert isinstance(result, Path)
    assert "claude" in str(result).lower() or result == Path("claude")


def test_register_mcp_claude_code_subprocess_error_falls_back(monkeypatch, tmp_path):
    """CalledProcessError triggers fallback to register_mcp(tool='claude')."""
    import subprocess

    fallback_calls = []

    def mock_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    def mock_register_mcp(**kwargs):
        fallback_calls.append(kwargs)
        return tmp_path / ".claude.json"

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr("jaxlint.skills.export.register_mcp", mock_register_mcp)

    result = register_mcp_claude_code(dry_run=False)

    assert len(fallback_calls) == 1
    assert fallback_calls[0]["tool"] == "claude"
    assert fallback_calls[0]["level"] == "user"
    assert not fallback_calls[0]["dry_run"]
    assert fallback_calls[0].get("_config_override") is None
    assert result == tmp_path / ".claude.json"


def test_register_mcp_claude_code_subprocess_absent_falls_back(monkeypatch, tmp_path):
    """FileNotFoundError (claude not on PATH) triggers fallback to register_mcp."""
    import subprocess

    fallback_calls = []

    def mock_run(args, **kwargs):
        raise FileNotFoundError("claude: not found")

    def mock_register_mcp(**kwargs):
        fallback_calls.append(kwargs)
        return tmp_path / ".claude.json"

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr("jaxlint.skills.export.register_mcp", mock_register_mcp)

    result = register_mcp_claude_code(dry_run=False)

    assert len(fallback_calls) == 1
    assert fallback_calls[0]["tool"] == "claude"
    assert result == tmp_path / ".claude.json"


# ---------------------------------------------------------------------------
# register_mcp_cursor
# ---------------------------------------------------------------------------

def test_register_mcp_cursor_dry_run(tmp_path):
    """dry_run=True returns path without writing."""
    cfg = tmp_path / ".mcp.json"
    result = register_mcp_cursor(dry_run=True, _config_override=cfg)
    assert result == cfg
    assert not cfg.exists()


def test_register_mcp_cursor_writes_config(tmp_path):
    """dry_run=False writes .mcp.json with mcpServers.jaxlint entry."""
    cfg = tmp_path / ".mcp.json"
    result = register_mcp_cursor(dry_run=False, _config_override=cfg)
    assert result == cfg
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert "mcpServers" in data
    assert "jaxlint" in data["mcpServers"]
    assert "command" in data["mcpServers"]["jaxlint"]


def test_register_mcp_cursor_merges_existing(tmp_path):
    """Preserves existing mcpServers entries."""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "cmd"}}}))
    register_mcp_cursor(dry_run=False, _config_override=cfg)
    data = json.loads(cfg.read_text())
    assert "other" in data["mcpServers"]
    assert "jaxlint" in data["mcpServers"]


def test_register_mcp_cursor_handles_corrupt_json(tmp_path):
    """Treats corrupt JSON as empty config, writes cleanly."""
    cfg = tmp_path / ".mcp.json"
    cfg.write_text("{ not valid json")
    register_mcp_cursor(dry_run=False, _config_override=cfg)
    data = json.loads(cfg.read_text())
    assert "jaxlint" in data["mcpServers"]


# ---------------------------------------------------------------------------
# setup command
# ---------------------------------------------------------------------------

def test_setup_cmd_default_calls_claude_code_and_gemini():
    """Default (no --tool): calls claude-code and gemini, not cursor."""
    from unittest.mock import patch

    with patch("jaxlint.cli.setup_command.register_mcp_claude_code") as m_cc, \
         patch("jaxlint.cli.setup_command.register_mcp") as m_reg, \
         patch("jaxlint.cli.setup_command.export_skill"), \
         patch("jaxlint.cli.setup_command.list_skills", return_value=[]):
        m_cc.return_value = Path.home() / ".claude.json"
        m_reg.return_value = Path.home() / ".gemini" / "settings.json"

        result = runner.invoke(main_app, ["setup", "--dry-run"])

        assert result.exit_code == 0
        m_cc.assert_called_once_with(dry_run=True)
        gemini_calls = [c for c in m_reg.call_args_list if c.kwargs.get("tool") == "gemini"]
        assert len(gemini_calls) == 1


def test_setup_cmd_cursor_only_skips_skills_and_registers_cursor():
    """--tool cursor: registers cursor, skips skills, skips claude-code."""
    from unittest.mock import patch

    with patch("jaxlint.cli.setup_command.register_mcp_cursor") as m_cur, \
         patch("jaxlint.cli.setup_command.register_mcp_claude_code") as m_cc, \
         patch("jaxlint.cli.setup_command.export_skill") as m_exp, \
         patch("jaxlint.cli.setup_command.list_skills", return_value=[]):
        m_cur.return_value = Path(".cursor") / "mcp.json"

        result = runner.invoke(main_app, ["setup", "--tool", "cursor", "--dry-run"])

        assert result.exit_code == 0
        m_cur.assert_called_once_with(dry_run=True)
        m_cc.assert_not_called()
        m_exp.assert_not_called()  # cursor surface has no skill export


def test_setup_cmd_no_mcp_skips_all_registration():
    """--no-mcp: skips all registration functions."""
    from unittest.mock import patch

    with patch("jaxlint.cli.setup_command.register_mcp_claude_code") as m_cc, \
         patch("jaxlint.cli.setup_command.register_mcp") as m_reg, \
         patch("jaxlint.cli.setup_command.register_mcp_cursor") as m_cur, \
         patch("jaxlint.cli.setup_command.export_skill"), \
         patch("jaxlint.cli.setup_command.list_skills", return_value=[]):

        result = runner.invoke(main_app, ["setup", "--no-mcp", "--dry-run"])

        assert result.exit_code == 0
        m_cc.assert_not_called()
        m_reg.assert_not_called()
        m_cur.assert_not_called()


def test_setup_cmd_unknown_tool_exits_nonzero():
    """Unknown surface name: exits with code 1."""
    result = runner.invoke(main_app, ["setup", "--tool", "vscode-unknown"])
    assert result.exit_code == 1


def test_setup_cmd_cursor_actually_writes_mcp_json(tmp_path):
    """Integration: cursor surface calls real register_mcp_cursor and writes file."""
    from unittest.mock import patch
    from jaxlint.skills.export import register_mcp_cursor as real_cursor

    cursor_config = tmp_path / "cursor_mcp.json"

    def cursor_with_override(**kwargs):
        return real_cursor(dry_run=kwargs.get("dry_run", False), _config_override=cursor_config)

    with patch("jaxlint.cli.setup_command.register_mcp_cursor", side_effect=cursor_with_override), \
         patch("jaxlint.cli.setup_command.export_skill"), \
         patch("jaxlint.cli.setup_command.list_skills", return_value=[]):

        result = runner.invoke(main_app, ["setup", "--tool", "cursor"])

        assert result.exit_code == 0
        assert cursor_config.exists()
        data = json.loads(cursor_config.read_text())
        assert "mcpServers" in data
        assert "jaxlint" in data["mcpServers"]
