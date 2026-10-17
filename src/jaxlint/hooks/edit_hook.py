"""PostToolUse hook: lint the one JAX file an agent just wrote or edited, feed NEW findings back.

Registered in ``.praxia/manifest.toml`` as ``[[plugin.hook_specs]]`` (event ``PostToolUse``,
matcher ``Write|Edit|MultiEdit``) and installed as the console script ``jaxlint-hook``.
Claude Code runs it after every matching tool call, so this module imports only the standard
library at load time. ``jaxlint.core`` (and with it the rule engine) is imported lazily, and only
after the cheap regex prefilter says the file imports JAX, equinox or jaxtyping.

Output contract (Claude Code hook JSON on stdout, exit 0 always):

* new findings, advisory: ``{"hookSpecificOutput": {"hookEventName": "PostToolUse",
  "additionalContext": "<text>"}}``
* new error-severity finding with ``hook-block-on-error = true`` in the project config:
  ``{"decision": "block", "reason": "<text>"}``

A finding is "new" when its identity (rule id, message, sha256 of its source line) was not
present at the previous check of the same file. Line numbers are not part of the identity, so
an edit that shifts lines does not re-announce old findings. The identities live in
``<cache-dir>/edit-hook/<sha256 of resolved path>.json``, where ``cache-dir`` is the project's
``cache-dir`` setting. With no cache dir configured there is no state, so an ``Edit`` or
``MultiEdit`` reports only findings on lines the edit wrote (``new_string``), and a ``Write``
reports every finding. The hook never picks a state location of its own.

Exit code 2 is never used: a block is a JSON decision, and any internal error is one line on
stderr with exit 0.

Telemetry: when ``CISTERNAL_TELEMETRY`` is truthy (cisternal's rule, copied in
``_telemetry_requested``), each hook run emits a ``jaxlint.hook.check`` span around the rule run and one
``jaxlint.hook`` span with the run's results, both through cisternal. Span fields carry only the sha256 of the resolved
path, never the path, the source or any message. When the variable is unset, cisternal is not
imported at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, TextIO

FORMAT_VERSION = 1
MAX_REPORTED = 20
STATE_SUBDIR = "edit-hook"

_JAX_IMPORT = re.compile(
    r"^\s*(import jax\b|from jax\b|import equinox\b|from equinox\b|import jaxtyping\b|from jaxtyping\b)",
    re.MULTILINE,
)

# Copy of cisternal's rule (src/cisternal/probe/telemetry_env.py:18-23, consumer_telemetry_enabled).
# Kept here so the disabled path imports nothing from cisternal.
_TELEMETRY_ENV = "CISTERNAL_TELEMETRY"
_TELEMETRY_TRUTHY = frozenset({"1", "true", "yes", "all"})
_TELEMETRY_CONSUMER = "jaxlint"


def _telemetry_requested() -> bool:
    raw = os.environ.get(_TELEMETRY_ENV, "").strip().lower()
    if not raw:
        return False
    return raw in _TELEMETRY_TRUTHY or raw == _TELEMETRY_CONSUMER


def _path_digest(source: Path) -> str:
    """sha256 of the resolved path: the only form of the path that telemetry ever carries."""
    return hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()


class _SafeSpan(AbstractContextManager):
    """Wraps a cisternal span so that a telemetry failure never reaches the hook.

    An exception raised by the hook's own body is re-raised unchanged (cisternal's span
    re-raises it too); any exception raised by cisternal is swallowed after one stderr line.
    """

    def __init__(self, owner: "_Telemetry", name: str, fields: dict[str, Any]) -> None:
        self._owner = owner
        self._name = name
        self._fields = fields
        self._cm: Any = None

    def __enter__(self) -> None:
        try:
            self._cm = self._owner.span_fn(self._name, **self._fields)
            self._cm.__enter__()
        except Exception as exc:  # noqa: BLE001 - telemetry must never affect the hook
            self._cm = None
            self._owner.note_disabled(exc)

    def __exit__(self, et: Any, ev: Any, tb: Any) -> bool:
        cm, self._cm = self._cm, None
        if cm is None:
            return False
        try:
            cm.__exit__(et, ev, tb)
        except Exception as exc:  # noqa: BLE001 - a body exception is re-raised by returning False
            if exc is not ev:
                self._owner.note_disabled(exc)
        return False


class _Telemetry:
    """cisternal spans for one hook run. Constructed only when ``CISTERNAL_TELEMETRY`` is set."""

    def __init__(self) -> None:
        self.span_fn: Any = None
        self._owns_pipeline = False
        self._noted = False
        try:
            from cisternal.telemetry.span import span
        except ImportError:
            return
        except Exception as exc:  # noqa: BLE001
            self.note_disabled(exc)
            return
        self.span_fn = span
        try:
            import cisternal

            if cisternal.get_pipeline() is None:
                cisternal.init()
                self._owns_pipeline = True
        except ImportError:
            self.span_fn = None
        except Exception as exc:  # noqa: BLE001
            self.span_fn = None
            self.note_disabled(exc)

    def note_disabled(self, exc: BaseException) -> None:
        if not self._noted:
            self._noted = True
            sys.stderr.write(f"jaxlint-hook: telemetry disabled: {type(exc).__name__}\n")

    def span(self, name: str, **fields: Any) -> AbstractContextManager[Any]:
        if self.span_fn is None:
            return nullcontext()
        return _SafeSpan(self, name, fields)

    def close(self) -> None:
        if not self._owns_pipeline:
            return
        try:
            from cisternal.telemetry.pipeline import shutdown_pipeline

            shutdown_pipeline()
        except Exception as exc:  # noqa: BLE001
            self.note_disabled(exc)


def _span(tel: _Telemetry | None, name: str, **fields: Any) -> AbstractContextManager[Any]:
    return nullcontext() if tel is None else tel.span(name, **fields)


def _read_target(stdin: TextIO) -> tuple[str, Path, set[str] | None] | None:
    """Return the tool name, the edited ``.py`` path and the stripped lines the edit wrote.

    The line set is None for a ``Write`` (the whole file is new) or an unrecognised payload
    shape. Returns None when there is nothing to lint.
    """
    try:
        payload = json.loads(stdin.read())
    except (ValueError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    tool_name = payload.get("tool_name")
    tool_name = tool_name if isinstance(tool_name, str) else ""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return None
    path = Path(file_path)
    if not path.is_file():
        return None
    written: list[str] = []
    if isinstance(tool_input.get("new_string"), str):
        written.append(tool_input["new_string"])
    edits = tool_input.get("edits")
    if isinstance(edits, list):
        written.extend(
            e["new_string"]
            for e in edits
            if isinstance(e, dict) and isinstance(e.get("new_string"), str)
        )
    if not written:
        return tool_name, path, None
    return (
        tool_name,
        path,
        {s for text in written for line in text.splitlines() if (s := line.strip())},
    )


def _identity(rule_id: str, message: str, line_text: str) -> tuple[str, str, str]:
    return (rule_id, message, hashlib.sha256(line_text.strip().encode("utf-8")).hexdigest())


def _line_text(lines: list[str], line: int) -> str:
    return lines[line - 1] if 1 <= line <= len(lines) else ""


def _state_path(cache_dir: Path, source: Path) -> Path:
    key = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()
    return cache_dir / STATE_SUBDIR / f"{key}.json"


def _load_previous(state_file: Path | None) -> Counter:
    if state_file is None:
        return Counter()
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        if int(data.get("format", -1)) != FORMAT_VERSION:
            return Counter()
        return Counter(tuple(item) for item in data["identities"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return Counter()


def _store_current(state_file: Path, identities: list[tuple[str, str, str]], source: Path) -> None:
    """Write the state atomically (tmp file in the same directory, then ``os.replace``)."""
    body = json.dumps(
        {
            "format": FORMAT_VERSION,
            "path": str(source.resolve()),
            "identities": [list(i) for i in identities],
        },
        sort_keys=True,
    )
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_name(state_file.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, state_file)
    finally:
        if tmp.exists():
            tmp.unlink()


def _format_text(source: Path, new: list[Any], followups: dict[str, str]) -> str:
    out: list[str] = []
    for d in new[:MAX_REPORTED]:
        out.append(f"{source}:{d.line} {d.rule_id} {d.severity.value}: {d.message}")
        followup = followups.get(d.rule_id)
        if followup:
            out.append(f"  → {followup}")
    if len(new) > MAX_REPORTED:
        out.append(f"({len(new) - MAX_REPORTED} more)")
    return "\n".join(out)


def _check(
    source: Path, written: set[str] | None, stdout: TextIO, tel: _Telemetry | None
) -> tuple[int, int, bool]:
    """Lint *source*, write any new findings to *stdout*.

    Returns ``(n_findings, n_new, blocked)`` for the outer telemetry span.
    """
    # Deferred imports: the rule engine is loaded only for a file that imports JAX.
    from jaxlint.core.config import load_config
    from jaxlint.core.rules import RULE_FOLLOWUPS
    from jaxlint.core.runner import run_checks
    from jaxlint.core.diagnostic import Severity

    cfg = load_config(source.resolve().parent)
    with _span(tel, "jaxlint.hook.check", path_sha256=_path_digest(source)):
        diags = run_checks([source], cfg=cfg, jobs=1, cache_dir=cfg.cache_dir)

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    identities = [_identity(d.rule_id, d.message, _line_text(lines, d.line)) for d in diags]

    state_file = _state_path(cfg.cache_dir, source) if cfg.cache_dir is not None else None
    previous = _load_previous(state_file)
    remaining = previous.copy()
    new: list[Any] = []
    for d, ident in zip(diags, identities):
        if state_file is None and written is not None:
            # Stateless: only lines this edit wrote are new.
            if _line_text(lines, d.line).strip() in written:
                new.append(d)
        elif remaining[ident] > 0:
            remaining[ident] -= 1
        else:
            new.append(d)

    if state_file is not None:
        try:
            _store_current(state_file, identities, source)
        except OSError:
            pass  # state is an optimisation; findings are still reported

    if not new:
        return len(diags), 0, False
    text = _format_text(source, new, RULE_FOLLOWUPS)
    blocked = cfg.hook_block_on_error and any(d.severity == Severity.ERROR for d in new)
    if blocked:
        payload: dict[str, Any] = {"decision": "block", "reason": text}
    else:
        payload = {
            "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text},
        }
    stdout.write(json.dumps(payload) + "\n")
    return len(diags), len(new), bool(blocked)


def _hook(stdin: TextIO, stdout: TextIO, tel: _Telemetry | None, facts: dict[str, Any]) -> None:
    target = _read_target(stdin)
    if target is None:
        return
    tool_name, source, written = target
    facts["tool_name"] = tool_name
    facts["path_sha256"] = _path_digest(source)
    try:
        raw = source.read_bytes()
    except OSError:
        return
    if not _JAX_IMPORT.search(raw.decode("utf-8", errors="replace")):
        return
    facts["jax_file"] = True
    facts["n_findings"], facts["n_new"], facts["blocked"] = _check(source, written, stdout, tel)


def _run(stdin: TextIO, stdout: TextIO) -> None:
    tel = _Telemetry() if _telemetry_requested() else None
    started = time.perf_counter()
    facts: dict[str, Any] = {
        "tool_name": "",
        "path_sha256": "",
        "jax_file": False,
        "n_findings": 0,
        "n_new": 0,
        "blocked": False,
    }
    try:
        _hook(stdin, stdout, tel, facts)
    finally:
        if tel is not None:
            # cisternal fixes a span's fields when it is entered, so the outer span is emitted
            # once the results are known; its elapsed_ms is measured here, not by the span.
            if facts["path_sha256"]:
                facts["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                with _span(tel, "jaxlint.hook", **facts):
                    pass
            tel.close()


def main() -> int:
    try:
        _run(sys.stdin, sys.stdout)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 - the hook must never exit 2 or crash the agent
        sys.stderr.write(f"jaxlint-hook: internal error: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
