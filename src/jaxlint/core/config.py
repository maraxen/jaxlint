"""Load [tool.jaxlint] from pyproject.toml."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


DEFAULT_REQUIRED_SECTIONS = ("Parameters", "Returns")
SECTION_ORDER_DEFAULT = ("Parameters", "Returns", "Raises", "Examples", "Pseudocode", "Hardware Notes")


@dataclass
class LintConfig:
    select: frozenset[str] = frozenset()
    ignore: frozenset[str] = frozenset()
    docstring_style: str = "numpy"
    required_sections: tuple[str, ...] = DEFAULT_REQUIRED_SECTIONS
    section_order: tuple[str, ...] = SECTION_ORDER_DEFAULT
    strict_shapes: bool = False
    rubric_threshold: float = 0.75
    per_file_ignores: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> LintConfig:
        return cls()


def load_config(start: Path | None = None) -> LintConfig:
    """Walk upward from ``start`` (or cwd) for pyproject.toml and parse [tool.jaxlint]."""
    root = start.resolve() if start else Path.cwd()
    for candidate in [root, *root.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            return _parse_tool_jaxlint(pyproject)
    return LintConfig.default()


def _parse_tool_jaxlint(path: Path) -> LintConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw: dict[str, Any] = (data.get("tool") or {}).get("jaxlint") or {}

    sel = raw.get("select")
    ign = raw.get("ignore")

    cfg = LintConfig(
        select=frozenset(sel) if sel else frozenset(),
        ignore=frozenset(ign) if ign else frozenset(),
        docstring_style=raw.get("docstring-style", "numpy"),
        required_sections=tuple(raw.get("required-sections") or DEFAULT_REQUIRED_SECTIONS),
        section_order=tuple(raw.get("section-order") or SECTION_ORDER_DEFAULT),
        strict_shapes=bool(raw.get("strict-shapes", False)),
        rubric_threshold=float(raw.get("rubric-threshold", 0.75)),
    )

    pf = raw.get("per-file-ignores") or {}
    cfg.per_file_ignores = {k: tuple(v) if isinstance(v, list) else (v,) for k, v in pf.items()}
    return cfg


def pattern_matches(rule_id: str, pattern: str) -> bool:
    if rule_id == pattern:
        return True
    prefix = pattern.rstrip("*")
    return rule_id.startswith(prefix)


def rule_enabled(rule_id: str, cfg: LintConfig, file_path: Path) -> bool:
    """Apply select/ignore and per-file-ignores (glob patterns)."""
    rel = str(file_path)

    def matches_any(pat_set: frozenset[str]) -> bool:
        if not pat_set:
            return False
        return any(pattern_matches(rule_id, p) for p in pat_set)

    for glob_pat, ignores in cfg.per_file_ignores.items():
        if fnmatch.fnmatch(rel, glob_pat):
            ign = ignores if isinstance(ignores, tuple) else (ignores,)
            if matches_any(frozenset(str(x) for x in ign)):
                return False

    if cfg.ignore and matches_any(cfg.ignore):
        return False
    if not cfg.select:
        return True
    return matches_any(cfg.select)


def starter_toml_snippet() -> str:
    return """[tool.jaxlint]
# select = ["JL", "JD", "JM"]
# ignore = ["JL005"]
docstring-style = "numpy"
required-sections = ["Parameters", "Returns"]
# section-order = ["Parameters", "Returns", "Examples"]
strict-shapes = false
rubric-threshold = 0.75

# [tool.jaxlint.per-file-ignores]
# "tests/**" = ["JL005"]
"""
