"""Load [tool.jaxlint] from jaxlint.toml or pyproject.toml."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, replace
from pathlib import Path
import tomllib
from typing import Any

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.rules import KNOWN_RULE_IDS

DEFAULT_REQUIRED_SECTIONS = ("Parameters", "Returns")
SECTION_ORDER_DEFAULT = ("Parameters", "Returns", "Raises", "Examples", "Pseudocode", "Hardware Notes")

_VALID_SEVERITY_TOKENS = frozenset(s.value for s in Severity)


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
    cache_dir: Path | None = None
    severity_overrides: dict[str, Severity] = field(default_factory=dict)

    @classmethod
    def default(cls) -> LintConfig:
        return cls()


def load_config(start: Path | None = None) -> LintConfig:
    """Load ``[tool.jaxlint]`` from the config discovery walk or a direct config file path.

    * ``start`` is ``None`` — walk upward from the current working directory.
    * ``start`` is a directory — walk upward from ``start.resolve()`` (``[root] + root.parents``).
    * ``start`` is a file — parse ``jaxlint.toml`` or ``pyproject.toml`` at that path (no walk);
      relative TOML paths resolve against the directory containing the file.
    """
    if start is None:
        return _load_config_walk(Path.cwd().resolve())

    path = start.resolve()
    if path.is_file():
        if path.name not in ("jaxlint.toml", "pyproject.toml"):
            msg = (
                f"Config file must be named 'jaxlint.toml' or 'pyproject.toml' (got {path.name!r}): {path}"
            )
            raise ValueError(msg)
        return _parse_tool_jaxlint(path)

    return _load_config_walk(path)


def _load_config_walk(root: Path) -> LintConfig:
    for candidate in [root, *root.parents]:
        jxl = candidate / "jaxlint.toml"
        pyr = candidate / "pyproject.toml"
        if jxl.is_file() and pyr.is_file():
            msg = f"Both jaxlint.toml and pyproject.toml exist in {candidate}; keep only one."
            raise ValueError(msg)
        if jxl.is_file():
            return _parse_tool_jaxlint(jxl)
        if pyr.is_file():
            return _parse_tool_jaxlint(pyr)
    return LintConfig.default()


def _parse_severity_value(raw: object, *, key: str, path: Path) -> Severity:
    if not isinstance(raw, str):
        msg = f"{path}: [tool.jaxlint] severity-overrides[{key!r}] must be a string, got {type(raw).__name__}"
        raise ValueError(msg)
    s = raw.strip().lower()
    if s not in _VALID_SEVERITY_TOKENS:
        msg = f"{path}: invalid severity {raw!r} for key {key!r} (expected error, warning, or info)"
        raise ValueError(msg)
    return Severity(s)


def _validate_severity_override_keys(keys: list[str], *, path: Path) -> None:
    for key in keys:
        if not any(pattern_matches(rule_id, key) for rule_id in KNOWN_RULE_IDS):
            msg = f"{path}: severity-overrides key {key!r} does not match any known rule id"
            raise ValueError(msg)


def _parse_severity_overrides_table(raw: dict[str, Any], *, path: Path) -> dict[str, Severity]:
    if not raw:
        return {}
    out: dict[str, Severity] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            msg = f"{path}: severity-overrides keys must be strings, got {type(k).__name__}"
            raise ValueError(msg)
        out[k] = _parse_severity_value(v, key=k, path=path)
    _validate_severity_override_keys(sorted(out.keys()), path=path)
    return out


def _parse_tool_jaxlint(path: Path) -> LintConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw: dict[str, Any] = (data.get("tool") or {}).get("jaxlint") or {}

    sel = raw.get("select")
    ign = raw.get("ignore")

    so_raw = raw.get("severity-overrides")
    if so_raw is not None and not isinstance(so_raw, dict):
        msg = f"{path}: [tool.jaxlint] severity-overrides must be a table, got {type(so_raw).__name__}"
        raise ValueError(msg)
    severity_overrides = _parse_severity_overrides_table(so_raw or {}, path=path)

    cfg = LintConfig(
        select=frozenset(sel) if sel else frozenset(),
        ignore=frozenset(ign) if ign else frozenset(),
        docstring_style=raw.get("docstring-style", "numpy"),
        required_sections=tuple(raw.get("required-sections") or DEFAULT_REQUIRED_SECTIONS),
        section_order=tuple(raw.get("section-order") or SECTION_ORDER_DEFAULT),
        strict_shapes=bool(raw.get("strict-shapes", False)),
        rubric_threshold=float(raw.get("rubric-threshold", 0.75)),
        severity_overrides=severity_overrides,
    )

    pf = raw.get("per-file-ignores") or {}
    cfg.per_file_ignores = {k: tuple(v) if isinstance(v, list) else (v,) for k, v in pf.items()}

    cd_raw = raw.get("cache-dir")
    if cd_raw:
        p = Path(str(cd_raw))
        cfg.cache_dir = p.resolve() if p.is_absolute() else (path.parent / p).resolve()
    return cfg


def pattern_matches(rule_id: str, pattern: str) -> bool:
    if rule_id == pattern:
        return True
    prefix = pattern.rstrip("*")
    return rule_id.startswith(prefix)


def _override_choice_sort_key(rule_id: str, pattern: str) -> tuple[int, int, str]:
    exact = 0 if pattern == rule_id else 1
    return (exact, -len(pattern.rstrip("*")), pattern)


def effective_severity(d: Diagnostic, cfg: LintConfig) -> Diagnostic:
    """Apply ``severity-overrides`` so configured severity always wins (after rule selection)."""
    if not cfg.severity_overrides:
        return d
    candidates = [k for k in cfg.severity_overrides if pattern_matches(d.rule_id, k)]
    if not candidates:
        return d
    best = min(candidates, key=lambda k: _override_choice_sort_key(d.rule_id, k))
    new_sev = cfg.severity_overrides[best]
    if new_sev is d.severity:
        return d
    return replace(d, severity=new_sev)


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

# cache-dir = ".jaxlint-cache"   # optional; resolved relative to the config file directory

# Per-rule severity (after sensors); overrides strict-shapes / default rule severity.
# [tool.jaxlint.severity-overrides]
# JL005 = "warning"

# [tool.jaxlint.per-file-ignores]
# "tests/**" = ["JL005"]
"""
