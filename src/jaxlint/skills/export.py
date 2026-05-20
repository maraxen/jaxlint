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
            content = content[:insert_at] + "\n" + stamp + content[insert_at:].lstrip("\n") + "\n"
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
