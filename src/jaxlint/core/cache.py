"""Opt-in disk cache for ``run_checks`` — serialized diagnostics only (stdlib)."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jaxlint.core as _core_pkg
from jaxlint import __version__
from jaxlint.core.config import LintConfig
from jaxlint.core.diagnostic import Diagnostic, Severity

CACHE_FORMAT_VERSION = 3


def _core_root() -> Path:
    """Directory of the installed ``jaxlint/core`` package (wheel or editable)."""
    return Path(_core_pkg.__file__).parent


def _hash_rule_sources(root: Path) -> str:
    """sha256 over sorted (relative posix path, bytes) of every ``.py`` under ``root``."""
    files = sorted(
        ((p.relative_to(root).as_posix(), p) for p in root.rglob("*.py") if p.is_file()),
        key=lambda t: t[0],
    )
    h = hashlib.sha256()
    for rel, p in files:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


@functools.cache
def rule_source_hash() -> str:
    """Hash of the rule implementation source, computed once per process.

    Keys the disk cache so that editing a rule (or any module feeding diagnostics)
    invalidates entries even when ``__version__`` is unchanged, as happens with
    editable installs.
    """
    return _hash_rule_sources(_core_root())


def _canonical_config_dict(cfg: LintConfig) -> dict[str, Any]:
    pf = cfg.per_file_ignores
    sorted_pf = {
        k: list(pf[k]) if isinstance(pf[k], tuple) else [pf[k]]
        for k in sorted(pf.keys(), key=lambda x: x)
    }
    return {
        "docstring_style": cfg.docstring_style,
        "ignore": sorted(cfg.ignore),
        "per_file_ignores": sorted_pf,
        "required_sections": list(cfg.required_sections),
        "rubric_threshold": cfg.rubric_threshold,
        "section_order": list(cfg.section_order),
        "select": sorted(cfg.select),
        "severity_overrides": {k: cfg.severity_overrides[k].value for k in sorted(cfg.severity_overrides.keys())},
        "strict_shapes": cfg.strict_shapes,
        "host_loop_jnp_ops": cfg.host_loop_jnp_ops,
    }


def namespace_segment(
    cfg: LintConfig,
    *,
    perf: bool,
    doc: bool,
    mathjx: bool,
) -> str:
    """Stable subdirectory name under ``cache_dir`` for this lint configuration."""
    payload = {
        "CACHE_FORMAT_VERSION": CACHE_FORMAT_VERSION,
        "cfg": _canonical_config_dict(cfg),
        "doc": doc,
        "jaxlint_version": __version__,
        "mathjx": mathjx,
        "perf": perf,
        "rule_source": rule_source_hash(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def content_digest(raw: bytes) -> str:
    """SHA-256 of a file's exact bytes; the cache key for its diagnostics."""
    return hashlib.sha256(raw).hexdigest()


def _entry_basename(py: Path, content_sha256: str) -> str:
    """Entries are keyed by resolved path and file *content*, never by mtime/size.

    An in-place edit that keeps size and mtime (or a restored mtime) must miss.
    """
    key = f"{py.resolve()}\0{content_sha256}".encode()
    return hashlib.sha256(key).hexdigest()


def _entry_paths(cache_root: Path, basename: str) -> tuple[Path, Path]:
    """Return (json_path, lock_path) with 2-char shard."""
    shard = basename[:2]
    rest = basename[2:]
    d = cache_root / "e" / shard
    return d / f"{rest}.json", d / f"{rest}.json.lock"


def _diagnostic_to_obj(d: Diagnostic) -> dict[str, Any]:
    return {
        "rule_id": d.rule_id,
        "severity": d.severity.value,
        "message": d.message,
        "path": str(d.path),
        "line": d.line,
        "col": d.col,
        "end_line": d.end_line,
        "end_col": d.end_col,
    }


def _diagnostic_from_obj(o: dict[str, Any]) -> Diagnostic:
    return Diagnostic(
        rule_id=o["rule_id"],
        severity=Severity(o["severity"]),
        message=o["message"],
        path=Path(o["path"]),
        line=int(o["line"]),
        col=int(o["col"]),
        end_line=o.get("end_line"),
        end_col=o.get("end_col"),
    )


def try_load_diagnostics(
    py: Path,
    *,
    cfg: LintConfig,
    perf: bool,
    doc: bool,
    mathjx: bool,
    cache_dir: Path,
    content_sha256: str,
) -> list[Diagnostic] | None:
    """Return cached diagnostics for exactly this content, else ``None``."""
    ns = namespace_segment(cfg, perf=perf, doc=doc, mathjx=mathjx)
    root = cache_dir / ns
    json_path, lock_path = _entry_paths(root, _entry_basename(py, content_sha256))

    if not json_path.is_file():
        return None

    data = _read_json_under_lock(lock_path, json_path)
    if data is None:
        return None

    try:
        if int(data.get("format", -1)) != CACHE_FORMAT_VERSION:
            return None
        if str(data.get("path")) != str(py.resolve()):
            return None
        if str(data.get("content_sha256")) != content_sha256:
            return None
        raw_diags = data["diagnostics"]
        if not isinstance(raw_diags, list):
            return None
        return [_diagnostic_from_obj(x) for x in raw_diags]
    except (KeyError, TypeError, ValueError):
        return None


def _read_json_under_lock(lock_path: Path, json_path: Path) -> dict[str, Any] | None:
    try:
        with _file_lock(lock_path, exclusive=False):
            raw = json_path.read_text(encoding="utf-8")
        return json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def store_diagnostics(
    py: Path,
    *,
    cfg: LintConfig,
    perf: bool,
    doc: bool,
    mathjx: bool,
    cache_dir: Path,
    content_sha256: str,
    diagnostics: list[Diagnostic],
) -> None:
    """Write cache entry; failures are silent (fail-open)."""
    ns = namespace_segment(cfg, perf=perf, doc=doc, mathjx=mathjx)
    root = cache_dir / ns
    json_path, lock_path = _entry_paths(root, _entry_basename(py, content_sha256))

    envelope = {
        "content_sha256": content_sha256,
        "diagnostics": [_diagnostic_to_obj(d) for d in diagnostics],
        "format": CACHE_FORMAT_VERSION,
        "path": str(py.resolve()),
    }
    body = json.dumps(envelope, separators=(",", ":"), sort_keys=True)

    try:
        root.mkdir(parents=True, exist_ok=True)
        shard_dir = json_path.parent
        shard_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    try:
        with _file_lock(lock_path, exclusive=True):
            tmp_path.write_text(body, encoding="utf-8")
            os.replace(tmp_path, json_path)
    except OSError:
        try:
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)  # py3.8+: py311 has missing_ok
        except OSError:
            pass


@contextmanager
def _file_lock(lock_path: Path, *, exclusive: bool):
    """BSD advisory lock when available; no-op on platforms without flock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        yield
        return

    import fcntl

    lock_path.touch(exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lf:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lf.fileno(), op)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
