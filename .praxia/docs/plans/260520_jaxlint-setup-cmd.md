# `jaxlint setup` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `jaxlint setup` top-level command that exports skill files and registers the MCP server across three tool surfaces (claude-code, gemini, cursor) in one shot.

**Architecture:** `export.py` stays library-pure — two new functions `register_mcp_claude_code` and `register_mcp_cursor` return `Path` and raise `ExportError`, never call `typer.echo`. A new `src/jaxlint/cli/setup_command.py` owns all CLI output and orchestrates the 3-surface routing. Tests consolidated in `tests/test_skill_export.py`.

**Tech Stack:** Python 3.11+, Typer, subprocess (claude mcp add), pathlib, json, importlib.resources

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/jaxlint/skills/export.py` | Add `register_mcp_claude_code()` and `register_mcp_cursor()` |
| Create | `src/jaxlint/cli/setup_command.py` | `setup_cmd` with 3-surface routing; all typer.echo output |
| Modify | `src/jaxlint/cli/main.py` | Import + `app.command("setup")(setup_cmd)` |
| Extend | `tests/test_skill_export.py` | New tests for both new functions + setup_cmd |
| Create | `docs/setup-command.md` | User-facing documentation |

---

## Task 1: Add `register_mcp_claude_code()` to export.py

**Files:**
- Modify: `src/jaxlint/skills/export.py`
- Extend: `tests/test_skill_export.py`

- [ ] **Step 1: Write failing tests — subprocess success case**

Append to `tests/test_skill_export.py`:

```python
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
```

- [ ] **Step 2: Write failing test — subprocess CalledProcessError fallback**

```python
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
    assert fallback_calls[0]["dry_run"] == False   # fallback only reachable when not dry_run
    assert fallback_calls[0].get("_config_override") is None
    assert result == tmp_path / ".claude.json"
```

- [ ] **Step 3: Write failing test — FileNotFoundError (claude binary absent) fallback**

```python
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
```

- [ ] **Step 4: Run tests to confirm they fail**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "register_mcp_claude_code" -v 2>&1 | head -20
```

Expected: `ImportError` or `NameError` — function doesn't exist yet.

- [ ] **Step 5: Implement `register_mcp_claude_code` in export.py**

Add after the existing `register_mcp()` function in `src/jaxlint/skills/export.py`:

```python
def register_mcp_claude_code(
    *,
    dry_run: bool,
    _config_override: Path | None = None,
) -> Path:
    """Register jaxlint-mcp via `claude mcp add jaxlint <cmd>`.

    When dry_run=True, returns the claude binary path without invoking subprocess.
    When dry_run=False, calls `claude mcp add jaxlint <cmd>` via subprocess.
    On FileNotFoundError or CalledProcessError, falls back to register_mcp(tool="claude")
    which merges into ~/.claude.json using the existing merge logic.

    Returns the claude binary path on success, or the fallback config path on failure.
    """
    import subprocess

    mcp_cmd = shutil.which("jaxlint-mcp") or "uv run --with jaxlint jaxlint-mcp"

    # dry_run: subprocess is never called; fallback is unreachable
    if dry_run:
        return Path(shutil.which("claude") or "claude")

    try:
        subprocess.run(
            ["claude", "mcp", "add", "jaxlint", mcp_cmd],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(shutil.which("claude") or "claude")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Delegate to existing merge logic — avoids duplicating the JSON write
        return register_mcp(
            tool="claude",
            level="user",
            dry_run=False,
            _config_override=_config_override,
        )
```

- [ ] **Step 6: Run tests**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "register_mcp_claude_code" -v
```

Expected: 3/3 pass.

- [ ] **Step 7: Commit**

```bash
git add src/jaxlint/skills/export.py tests/test_skill_export.py
git commit -m "feat(export): add register_mcp_claude_code with subprocess and fallback"
```

---

## Task 2: Add `register_mcp_cursor()` to export.py

**Files:**
- Modify: `src/jaxlint/skills/export.py`
- Extend: `tests/test_skill_export.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_skill_export.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "register_mcp_cursor" -v 2>&1 | head -20
```

Expected: `ImportError` or `NameError`.

- [ ] **Step 3: Implement `register_mcp_cursor` in export.py**

Add after `register_mcp_claude_code()`:

```python
def register_mcp_cursor(
    *,
    dry_run: bool,
    _config_override: Path | None = None,
) -> Path:
    """Register jaxlint-mcp in Cursor's workspace MCP config (.cursor/mcp.json).

    Cursor reads MCP servers from .cursor/mcp.json in the workspace root.
    Merges into existing config; creates file if absent.

    Returns the config path (written or would-be-written).
    """
    cfg_path = _config_override if _config_override is not None else Path(".cursor") / "mcp.json"
    mcp_cmd = shutil.which("jaxlint-mcp") or "uv run --with jaxlint jaxlint-mcp"

    if not dry_run:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        existing.setdefault("mcpServers", {})["jaxlint"] = {"command": mcp_cmd}
        cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return cfg_path
```

- [ ] **Step 4: Run tests**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "register_mcp_cursor" -v
```

Expected: 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaxlint/skills/export.py tests/test_skill_export.py
git commit -m "feat(export): add register_mcp_cursor for Cursor workspace MCP config"
```

---

## Task 3: Create `setup_command.py` and wire into `main.py`

**Files:**
- Create: `src/jaxlint/cli/setup_command.py`
- Modify: `src/jaxlint/cli/main.py`
- Extend: `tests/test_skill_export.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_skill_export.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "setup_cmd" -v 2>&1 | head -20
```

Expected: `No such command 'setup'` or `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/jaxlint/cli/setup_command.py`**

```python
"""Typer command for `jaxlint setup` — unified MCP + skill onboarding."""

from __future__ import annotations

from pathlib import Path

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

    See docs/setup-command.md for full documentation.
    """
    # Deduplicate while preserving order
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
                typer.echo(
                    f"Warning: MCP registration failed for {surface}: {e}", err=True
                )
                errors.append(f"mcp@{surface}")

    if errors:
        raise typer.Exit(1)
```

- [ ] **Step 4: Wire into `src/jaxlint/cli/main.py`**

After the existing `from jaxlint.cli.skill_commands import skill_app` import (line 18), add:

```python
from jaxlint.cli.setup_command import setup_cmd
```

After `app.add_typer(skill_app, name="skill")` (line 21), add:

```python
app.command("setup")(setup_cmd)
```

- [ ] **Step 5: Run tests**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "setup_cmd" -v
```

Expected: 4/4 pass.

- [ ] **Step 6: Smoke test**

```bash
uv run jaxlint setup --help
uv run jaxlint setup --dry-run
uv run jaxlint setup --tool cursor --dry-run
```

Expected: `--dry-run` shows 6 skill export lines + 2 MCP registration lines; cursor shows only 1 MCP line (no skills).

- [ ] **Step 7: Commit**

```bash
git add src/jaxlint/cli/setup_command.py src/jaxlint/cli/main.py tests/test_skill_export.py
git commit -m "feat(cli): add 'jaxlint setup' command with claude-code, gemini, cursor surfaces"
```

---

## Task 4: Full test suite verification

**Files:** None — verification only.

- [ ] **Step 1: Run full suite**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: All tests pass, no regressions.

- [ ] **Step 2: Coverage check**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py --cov=jaxlint.skills --cov=jaxlint.cli.setup_command --cov-report=term-missing
```

Expected: ≥ 80% on `export.py` and `setup_command.py`.

- [ ] **Step 3: Commit if clean**

```bash
git status  # should be clean
```

---

## Task 5: Documentation

**Files:**
- Create: `docs/setup-command.md`

- [ ] **Step 1: Create docs file**

```markdown
# jaxlint setup

One-shot onboarding command that exports skill files and registers the MCP server
across your tools.

## Usage

```bash
jaxlint setup                                           # Default: claude-code + gemini
jaxlint setup --tool cursor                             # Cursor only
jaxlint setup --tool claude-code --tool cursor          # Multiple surfaces
jaxlint setup --dry-run                                 # Preview without writing
jaxlint setup --no-skills                               # MCP registration only
jaxlint setup --no-mcp                                  # Skill export only
```

## Surfaces

| Surface | Skills | MCP registration |
|---------|--------|-----------------|
| `claude-code` | `~/.claude/skills/{name}/SKILL.md` | `claude mcp add jaxlint <cmd>` (falls back to `~/.claude.json`) |
| `gemini` | `~/.gemini/skills/{name}/SKILL.md` | `~/.gemini/settings.json` |
| `cursor` | — | `.cursor/mcp.json` in workspace root |

## Failure behaviour

Surface failures print a warning and continue; other surfaces are not affected.
Exit 0 if all surfaces succeeded, exit 1 if any failed.

## Idempotent

Re-running `jaxlint setup` is safe — skill files are overwritten with the current
version-stamped content; MCP config entries are merged (existing entries preserved).
```

- [ ] **Step 2: Commit**

```bash
git add docs/setup-command.md
git commit -m "docs: add setup command documentation"
```

---

## Self-Review

**Spec coverage:**
- ✅ `jaxlint setup` with `--tool`, `--dry-run`, `--no-mcp`, `--no-skills` — Task 3
- ✅ `claude-code` surface via `claude mcp add` + skills — Tasks 1 + 3
- ✅ `gemini` surface via JSON merge + skills — Task 3 (reuses existing `register_mcp`)
- ✅ `cursor` surface via `.cursor/mcp.json`, no skills — Tasks 2 + 3
- ✅ Graceful per-surface failure + exit codes — Task 3
- ✅ Library-pure `export.py` (no typer) — Tasks 1 + 2
- ✅ All tests in `test_skill_export.py` — Tasks 1 + 2 + 3
- ✅ Documentation — Task 5

**Type consistency:**
- `register_mcp_claude_code(*, dry_run, _config_override=None) -> Path` ✅
- `register_mcp_cursor(*, dry_run, _config_override=None) -> Path` ✅
- `setup_cmd(tool: list[str], dry_run, no_mcp, no_skills) -> None` ✅
- All registration functions return `Path`, matching existing `register_mcp` ✅

**Placeholder scan:** No TBDs, no "implement later". All code blocks complete. ✅
