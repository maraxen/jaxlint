"""Load lint settings from ``jaxlint.toml`` or ``pyproject.toml``.

``jaxlint.toml`` accepts a first-class ``[jaxlint]`` table or legacy ``[tool.jaxlint]``
(but not both). ``pyproject.toml`` must use ``[tool.jaxlint]`` only — a top-level
``[jaxlint]`` table is rejected.
"""

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
    host_loop_jnp_ops: bool = False
    host_sync_everywhere: bool = False
    rubric_threshold: float = 0.75
    per_file_ignores: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cache_dir: Path | None = None
    severity_overrides: dict[str, Severity] = field(default_factory=dict)
    sarif_rule_docs_base: str | None = None

    @classmethod
    def default(cls) -> LintConfig:
        return cls()


def load_config(start: Path | None = None) -> LintConfig:
    """Load lint configuration from the discovery walk or a direct config file path.

    ``jaxlint.toml`` may use native ``[jaxlint]`` or legacy ``[tool.jaxlint]`` (mutually
    exclusive). ``pyproject.toml`` only allows ``[tool.jaxlint]``.

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


def _parse_severity_value(
    raw: object, *, key: str, path: Path, section_label: str
) -> Severity:
    if not isinstance(raw, str):
        msg = (
            f"{path}: {section_label} severity-overrides[{key!r}] must be a string, "
            f"got {type(raw).__name__}"
        )
        raise ValueError(msg)
    s = raw.strip().lower()
    if s not in _VALID_SEVERITY_TOKENS:
        msg = f"{path}: invalid severity {raw!r} for key {key!r} (expected error, warning, or info)"
        raise ValueError(msg)
    return Severity(s)


def _validate_severity_override_keys(keys: list[str], *, path: Path, section_label: str) -> None:
    for key in keys:
        if not any(pattern_matches(rule_id, key) for rule_id in KNOWN_RULE_IDS):
            msg = (
                f"{path}: {section_label} severity-overrides key {key!r} "
                "does not match any known rule id"
            )
            raise ValueError(msg)


def _parse_severity_overrides_table(
    raw: dict[str, Any], *, path: Path, section_label: str
) -> dict[str, Severity]:
    if not raw:
        return {}
    out: dict[str, Severity] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            msg = (
                f"{path}: {section_label} severity-overrides keys must be strings, "
                f"got {type(k).__name__}"
            )
            raise ValueError(msg)
        out[k] = _parse_severity_value(v, key=k, path=path, section_label=section_label)
    _validate_severity_override_keys(sorted(out.keys()), path=path, section_label=section_label)
    return out


def _extract_raw_jaxlint(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Return the jaxlint settings table after applying file-kind rules (see module docstring)."""
    raw, _section_label = _resolve_lint_raw_table(path, data)
    return raw


def _resolve_lint_raw_table(path: Path, data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Resolve the lint config dict and TOML section label used in parse-time errors."""
    name = path.name
    tool_obj = data.get("tool")
    tool: dict[str, Any] = tool_obj if isinstance(tool_obj, dict) else {}
    has_tool_jaxlint = "jaxlint" in tool
    has_root_jaxlint = "jaxlint" in data

    if name == "pyproject.toml":
        if has_root_jaxlint:
            msg = (
                f"{path}: configure jaxlint under [tool.jaxlint] only in pyproject.toml; "
                "remove the top-level [jaxlint] table."
            )
            raise ValueError(msg)
        jaxlint_raw = tool.get("jaxlint")
        if jaxlint_raw is None:
            return {}, "[tool.jaxlint]"
        if not isinstance(jaxlint_raw, dict):
            msg = (
                f"{path}: [tool.jaxlint] must be a table, "
                f"got {type(jaxlint_raw).__name__}"
            )
            raise ValueError(msg)
        return jaxlint_raw, "[tool.jaxlint]"

    if name == "jaxlint.toml":
        if has_root_jaxlint and has_tool_jaxlint:
            msg = (
                f"{path}: jaxlint.toml cannot contain both [jaxlint] and [tool.jaxlint]; "
                "use only one section."
            )
            raise ValueError(msg)
        if has_root_jaxlint:
            native = data["jaxlint"]
            if not isinstance(native, dict):
                msg = f"{path}: [jaxlint] must be a table, got {type(native).__name__}"
                raise ValueError(msg)
            return native, "[jaxlint]"
        if has_tool_jaxlint:
            legacy = tool["jaxlint"]
            if not isinstance(legacy, dict):
                msg = (
                    f"{path}: [tool.jaxlint] must be a table, "
                    f"got {type(legacy).__name__}"
                )
                raise ValueError(msg)
            return legacy, "[tool.jaxlint]"
        return {}, "[jaxlint]"

    msg = f"Config file must be named 'jaxlint.toml' or 'pyproject.toml' (got {name!r}): {path}"
    raise ValueError(msg)


def _parse_sarif_rule_docs_base(
    raw: object, *, path: Path, section_label: str
) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        msg = (
            f"{path}: {section_label} sarif-rule-docs-base must be a string, "
            f"got {type(raw).__name__}"
        )
        raise ValueError(msg)
    s = raw.strip()
    if not s:
        msg = f"{path}: {section_label} sarif-rule-docs-base must not be empty when set"
        raise ValueError(msg)
    if not s.startswith("https://"):
        msg = (
            f"{path}: {section_label} sarif-rule-docs-base must be an https:// URL "
            f"(got {raw!r})"
        )
        raise ValueError(msg)
    return s


def _lint_config_from_raw(raw: dict[str, Any], path: Path, *, section_label: str) -> LintConfig:
    sel = raw.get("select")
    ign = raw.get("ignore")

    so_raw = raw.get("severity-overrides")
    if so_raw is not None and not isinstance(so_raw, dict):
        msg = (
            f"{path}: {section_label} severity-overrides must be a table, "
            f"got {type(so_raw).__name__}"
        )
        raise ValueError(msg)
    severity_overrides = _parse_severity_overrides_table(
        so_raw or {}, path=path, section_label=section_label
    )

    srdb_raw = raw.get("sarif-rule-docs-base")
    sarif_rule_docs_base = _parse_sarif_rule_docs_base(
        srdb_raw, path=path, section_label=section_label
    )

    cfg = LintConfig(
        select=frozenset(sel) if sel else frozenset(),
        ignore=frozenset(ign) if ign else frozenset(),
        docstring_style=raw.get("docstring-style", "numpy"),
        required_sections=tuple(raw.get("required-sections") or DEFAULT_REQUIRED_SECTIONS),
        section_order=tuple(raw.get("section-order") or SECTION_ORDER_DEFAULT),
        strict_shapes=bool(raw.get("strict-shapes", False)),
        host_loop_jnp_ops=bool(raw.get("host-loop-jnp-ops", False)),
        host_sync_everywhere=bool(raw.get("host-sync-everywhere", False)),
        rubric_threshold=float(raw.get("rubric-threshold", 0.75)),
        severity_overrides=severity_overrides,
        sarif_rule_docs_base=sarif_rule_docs_base,
    )

    pf = raw.get("per-file-ignores") or {}
    cfg.per_file_ignores = {k: tuple(v) if isinstance(v, list) else (v,) for k, v in pf.items()}

    cd_raw = raw.get("cache-dir")
    if cd_raw:
        p = Path(str(cd_raw))
        cfg.cache_dir = p.resolve() if p.is_absolute() else (path.parent / p).resolve()
    return cfg


def _parse_tool_jaxlint(path: Path) -> LintConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw, section_label = _resolve_lint_raw_table(path, data)
    return _lint_config_from_raw(raw, path, section_label=section_label)


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
    return """[jaxlint]
# select = ["JL", "JD", "JM"]
# ignore = ["JL005"]
docstring-style = "numpy"
required-sections = ["Parameters", "Returns"]
# section-order = ["Parameters", "Returns", "Examples"]
strict-shapes = false
rubric-threshold = 0.75

# cache-dir = ".jaxlint-cache"   # optional; resolved relative to the config file directory

# Per-rule severity (after sensors); overrides strict-shapes / default rule severity.
# [jaxlint.severity-overrides]
# JL005 = "warning"

# [jaxlint.per-file-ignores]
# "tests/**" = ["JL005"]

# Optional SARIF help links: rule helpUri = base + "/" + rule_id (HTTPS only).
# sarif-rule-docs-base = "https://example.com/docs/rules"
"""
