# Skill Export CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `jaxlint skill` sub-command group that lists and exports the six JAX skill files to Claude/Gemini tool directories, following the same pattern as `bth export` in the bathos project.

**Architecture:** A new `src/jaxlint/skills/export.py` module handles skill discovery, source resolution (editable vs installed), version stamping, target path resolution, and MCP server registration. A Typer sub-app in `src/jaxlint/cli/skill_commands.py` exposes `jaxlint skill list` and `jaxlint skill export`. The six `SKILL.md` files under `agent_assets/skills/` are bundled into the wheel via `pyproject.toml` `force-include`.

**Tech Stack:** Python 3.11+, Typer, `importlib.resources`, `re` (frontmatter parsing), `json` (MCP config merge), `pathlib`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Bundle `agent_assets/` into wheel |
| Create | `src/jaxlint/skills/__init__.py` | Package marker |
| Create | `src/jaxlint/skills/export.py` | Core: discovery, resolve, stamp, export, MCP register |
| Create | `src/jaxlint/cli/skill_commands.py` | Typer sub-app: `list`, `export` |
| Modify | `src/jaxlint/cli/main.py` | Wire sub-app: `app.add_typer(skill_app, name="skill")` |
| Create | `tests/test_skill_export.py` | Unit + integration tests |

---

## Task 1: Bundle agent_assets into the wheel

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add force-include to pyproject.toml**

In `pyproject.toml`, after the existing `[tool.hatch.build.targets.wheel]` block (line 61), add:

```toml
[tool.hatch.build.targets.wheel.force-include]
"agent_assets" = "jaxlint/agent_assets"
```

This causes hatchling to embed `agent_assets/` at `jaxlint/agent_assets/` inside the installed wheel, making `importlib.resources.files("jaxlint") / "agent_assets"` resolve correctly in both editable and installed modes.

- [ ] **Step 2: Verify the current agent_assets directory exists**

```bash
ls /home/marielle/projects/jaxlint/agent_assets/skills/
```

Expected output: six directories — `developing-composable-jax`, `exporting-jax`, `using-equinox`, `using-jax`, `using-jaxlint`, `using-orbax`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bundle agent_assets into wheel for skill export"
```

---

## Task 2: Core export module

**Files:**
- Create: `src/jaxlint/skills/__init__.py`
- Create: `src/jaxlint/skills/export.py`
- Test: `tests/test_skill_export.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_skill_export.py`:

```python
"""Tests for jaxlint.skills.export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert len(skills) == 6
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -v 2>&1 | head -30
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'jaxlint.skills'`

- [ ] **Step 3: Create the package marker**

Create `src/jaxlint/skills/__init__.py`:

```python
"""Skill export utilities for jaxlint."""
```

- [ ] **Step 4: Implement `src/jaxlint/skills/export.py`**

Create `src/jaxlint/skills/export.py`:

```python
"""Export jaxlint skill files to Claude/Gemini tool directories."""

from __future__ import annotations

import importlib.resources
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jaxlint
from jaxlint import __version__

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_FM_KEY_RE = re.compile(r"^(\w[\w-]*):\s*(.+)$", re.MULTILINE)

# Target path templates — {name} is the skill name
_CLAUDE_TARGETS: dict[str, str] = {
    "user": str(Path.home() / ".claude" / "skills" / "{name}" / "SKILL.md"),
    "workspace": str(Path(".claude") / "skills" / "{name}" / "SKILL.md"),
}
_GEMINI_TARGETS: dict[str, str] = {
    "user": str(Path.home() / ".gemini" / "skills" / "{name}" / "SKILL.md"),
    "workspace": str(Path(".gemini") / "skills" / "{name}" / "SKILL.md"),
}

# MCP config file paths (default; overridable in tests via _config_override)
_CLAUDE_MCP_CONFIG: dict[str, Path] = {
    "user": Path.home() / ".claude.json",
    "workspace": Path(".claude") / "mcp.json",
}
_GEMINI_MCP_CONFIG: dict[str, Path] = {
    "user": Path.home() / ".gemini" / "settings.json",
    "workspace": Path(".gemini") / "settings.json",
}


class ExportError(Exception):
    """Raised when skill export fails."""


@dataclass(frozen=True)
class SkillInfo:
    """Metadata parsed from a skill's YAML frontmatter."""

    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class ExportResult:
    """Result of a single skill export operation."""

    skill_name: str
    target: Path
    dry_run: bool


def get_skills_root() -> Path:
    """Locate the agent_assets/skills/ directory (editable or installed)."""
    package_dir = Path(jaxlint.__file__).parent
    # Editable install: repo layout
    candidate = package_dir.parent.parent / "agent_assets" / "skills"
    if candidate.exists():
        return candidate
    # Installed wheel: bundled at jaxlint/agent_assets/skills/
    try:
        ref = importlib.resources.files("jaxlint") / "agent_assets" / "skills"
        with importlib.resources.as_file(ref) as p:
            if Path(p).exists():
                return Path(p)
    except Exception:
        pass
    raise ExportError(
        "Cannot find bundled skill assets. "
        "Re-install jaxlint or run from the repository root."
    )


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    return dict(_FM_KEY_RE.findall(m.group(1)))


def list_skills() -> list[SkillInfo]:
    """Return metadata for all available skills, sorted by name."""
    root = get_skills_root()
    skills: list[SkillInfo] = []
    for skill_dir in sorted(root.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not (skill_dir.is_dir() and skill_md.exists()):
            continue
        fm = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        name = fm.get("name", skill_dir.name)
        description = fm.get("description", "").strip('"').strip("'")
        skills.append(SkillInfo(name=name, description=description, path=skill_md))
    return skills


def _known_skill_names() -> set[str]:
    return {s.name for s in list_skills()}


def resolve_target(skill_name: str, *, tool: str, level: str) -> Path:
    """Return the install path for a skill given tool and level."""
    known = _known_skill_names()
    if skill_name not in known:
        raise ExportError(
            f"Unknown skill: {skill_name!r}. Available: {sorted(known)}"
        )
    if tool == "claude":
        templates = _CLAUDE_TARGETS
    elif tool == "gemini":
        templates = _GEMINI_TARGETS
    else:
        raise ExportError(f"Unknown tool: {tool!r}. Choose 'claude' or 'gemini'.")
    if level not in templates:
        raise ExportError(
            f"Unknown level: {level!r}. Choose {list(templates)!r}."
        )
    return Path(templates[level].format(name=skill_name))


def export_skill(
    skill_name: str,
    *,
    target: Path,
    dry_run: bool,
) -> ExportResult:
    """Write a skill's SKILL.md to *target*, inserting a version stamp."""
    # Validate skill name
    skills_by_name = {s.name: s for s in list_skills()}
    if skill_name not in skills_by_name:
        raise ExportError(
            f"Unknown skill: {skill_name!r}. Available: {sorted(skills_by_name)}"
        )

    if not dry_run:
        source = skills_by_name[skill_name].path
        content = source.read_text(encoding="utf-8")
        # Insert version stamp after closing --- of frontmatter
        stamp = (
            f"<!-- jaxlint v{__version__} | "
            f"exported {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} -->\n"
        )
        fm_match = _FRONTMATTER_RE.match(content)
        if fm_match:
            insert_at = fm_match.end()
            # Ensure we're after the newline that closes ---
            content = content[:insert_at] + "\n" + stamp + content[insert_at:]
        else:
            content = stamp + content

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return ExportResult(skill_name=skill_name, target=target, dry_run=dry_run)


def register_mcp(
    *,
    tool: str,
    level: str,
    dry_run: bool,
    _config_override: Path | None = None,
) -> Path:
    """Register jaxlint-mcp in the tool's MCP config file.

    Returns the config file path (written or would-be-written).
    """
    if tool == "claude":
        cfg_paths = _CLAUDE_MCP_CONFIG
    elif tool == "gemini":
        cfg_paths = _GEMINI_MCP_CONFIG
    else:
        raise ExportError(f"Unknown tool: {tool!r}.")

    if level not in cfg_paths:
        raise ExportError(f"Unknown level: {level!r}.")

    cfg_path = _config_override if _config_override is not None else cfg_paths[level]

    # Resolve jaxlint-mcp command
    mcp_cmd = shutil.which("jaxlint-mcp") or "uv run --with jaxlint jaxlint-mcp"
    entry = {"command": mcp_cmd}

    if not dry_run:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        existing.setdefault("mcpServers", {})["jaxlint"] = entry
        cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return cfg_path
```

- [ ] **Step 5: Run tests**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/jaxlint/skills/__init__.py src/jaxlint/skills/export.py tests/test_skill_export.py
git commit -m "feat(skills): core export module with list, resolve, export, register_mcp"
```

---

## Task 3: Skill CLI sub-app

**Files:**
- Create: `src/jaxlint/cli/skill_commands.py`
- Test: `tests/test_skill_export.py` (extend)

- [ ] **Step 1: Write failing CLI tests** — append to `tests/test_skill_export.py`:

```python
# ---------------------------------------------------------------------------
# CLI tests (append to existing test file)
# ---------------------------------------------------------------------------
from typer.testing import CliRunner
from jaxlint.cli.main import app as main_app

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


def test_skill_export_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXLINT_SKILL_TARGET_OVERRIDE", str(tmp_path))
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
    assert result.exit_code != 0


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
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -k "cli or skill_list or skill_export" -v 2>&1 | head -20
```

Expected: Fail — `skill` subcommand not found.

- [ ] **Step 3: Implement `src/jaxlint/cli/skill_commands.py`**

```python
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
```

- [ ] **Step 4: Wire sub-app into `src/jaxlint/cli/main.py`**

Add these two lines to `src/jaxlint/cli/main.py`:

After the existing imports block (around line 11), add:
```python
from jaxlint.cli.skill_commands import skill_app
```

After `app = typer.Typer(...)` (line 19), add:
```python
app.add_typer(skill_app, name="skill")
```

- [ ] **Step 5: Run CLI tests**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Smoke test the commands manually**

```bash
cd /home/marielle/projects/jaxlint && uv run jaxlint skill list
uv run jaxlint skill export --all --tool claude --level user --dry-run
```

Expected for `list`:
```
  developing-composable-jax  Use when Equinox __call__ methods are growing...
  exporting-jax              Use when exporting a JAX model to StableHLO...
  using-equinox              Use when defining eqx.Module subclasses...
  using-jax                  Use when writing or modifying JAX code...
  using-jaxlint              Use when running jaxlint checks...
  using-orbax                Use when saving/restoring model checkpoints...
```

Expected for dry-run export: six "Dry run — would export" lines + one "Dry run — would register jaxlint-mcp" line.

- [ ] **Step 7: Commit**

```bash
git add src/jaxlint/cli/skill_commands.py src/jaxlint/cli/main.py tests/test_skill_export.py
git commit -m "feat(cli): add 'jaxlint skill list' and 'jaxlint skill export' subcommands"
```

---

## Task 4: Move jax_dev skill from jaxbeans and clean up

**Files:**
- Verify: `/home/marielle/projects/jaxlint/agent_assets/skills/using-jax/SKILL.md` (already moved)
- Modify: `/home/marielle/projects/jaxbeans/agent_assets/skills/jax_dev/SKILL.md` — replace with redirect
- Modify: `/home/marielle/projects/jaxbeans/agent_assets/skills/jax_dev/sensors/graph_audit.py` — no change needed (still re-exports from jaxlint)

- [ ] **Step 1: Verify using-jax is present in jaxlint**

```bash
head -5 /home/marielle/projects/jaxlint/agent_assets/skills/using-jax/SKILL.md
```

Expected: YAML frontmatter with `name: using-jax`.

- [ ] **Step 2: Replace jaxbeans jax_dev skill with a redirect stub**

Overwrite `/home/marielle/projects/jaxbeans/agent_assets/skills/jax_dev/SKILL.md`:

```markdown
---
name: jax-dev-redirect
description: Redirects to using-jax in jaxlint. Install jaxlint skills with `jaxlint skill export --all`.
---

# JAX Dev Skill — Moved

This skill has moved to `jaxlint`. Install the full JAX skill suite:

```bash
jaxlint skill export --all --tool claude --level user
```

Skills installed:
- `using-jax` — JAX fundamentals
- `using-equinox` — Equinox modules and transforms
- `using-orbax` — Orbax checkpointing
- `using-jaxlint` — jaxlint usage
- `developing-composable-jax` — Equinox composability patterns
- `exporting-jax` — StableHLO, IREE, WebGPU export
```

- [ ] **Step 3: Commit in jaxbeans**

```bash
cd /home/marielle/projects/jaxbeans && git add agent_assets/skills/jax_dev/SKILL.md
git commit -m "chore(skills): replace jax_dev with redirect to jaxlint skill suite"
```

---

## Task 5: Verify full test suite and coverage

**Files:**
- None — verification only

- [ ] **Step 1: Run full jaxlint test suite**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All tests pass. No regressions.

- [ ] **Step 2: Check coverage for new code**

```bash
cd /home/marielle/projects/jaxlint && uv run pytest tests/test_skill_export.py --cov=jaxlint.skills --cov-report=term-missing
```

Expected: Coverage ≥ 85% on `jaxlint/skills/export.py`.

- [ ] **Step 3: Smoke test installed export for real**

```bash
cd /home/marielle/projects/jaxlint && uv run jaxlint skill export using-jax --tool claude --level workspace --dry-run
```

Expected:
```
Dry run — would export 'using-jax'  → .claude/skills/using-jax/SKILL.md
Dry run — would register jaxlint-mcp in ~/.claude.json
```

- [ ] **Step 4: Commit any remaining cleanup**

```bash
cd /home/marielle/projects/jaxlint
git add -A
git status  # verify nothing unintended
git commit -m "feat(skills): complete skill export CLI — list, export, MCP registration"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `jaxlint skill list` — Task 3
- ✅ `jaxlint skill export [name]` — Task 3
- ✅ `jaxlint skill export --all` — Task 3
- ✅ `--tool claude|gemini` — Task 3
- ✅ `--level user|workspace` — Task 3
- ✅ `--dry-run` — Tasks 2 + 3
- ✅ MCP server registration — Task 2
- ✅ Version stamp in exported file — Task 2
- ✅ Package bundling (wheel) — Task 1
- ✅ Move jax_dev from jaxbeans — Task 4
- ✅ Tests for all core functions — Tasks 2 + 3

**Type consistency check:**
- `list_skills()` returns `list[SkillInfo]` — consistent across Task 2 implementation and Task 3 usage ✅
- `export_skill(skill_name, *, target, dry_run)` — consistent across tests and implementation ✅
- `resolve_target(skill_name, *, tool, level)` — consistent ✅
- `register_mcp(*, tool, level, dry_run, _config_override)` — consistent ✅
- `ExportResult.dry_run: bool` — used in tests as `result.dry_run is True` ✅
- `SkillInfo.name`, `.description`, `.path` — all referenced in CLI and tests ✅

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N" shortcuts. All code blocks are complete. ✅
