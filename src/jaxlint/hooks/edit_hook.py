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
``cache-dir`` setting; with no cache dir configured every finding counts as new.

Exit code 2 is never used: a block is a JSON decision, and any internal error is one line on
stderr with exit 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

FORMAT_VERSION = 1
MAX_REPORTED = 20
STATE_SUBDIR = "edit-hook"

_JAX_IMPORT = re.compile(
    r"^\s*(import jax\b|from jax\b|import equinox\b|from equinox\b|import jaxtyping\b|from jaxtyping\b)",
    re.MULTILINE,
)


def _read_target(stdin: TextIO) -> Path | None:
    """Return the edited ``.py`` path from the hook payload, or None when there is nothing to lint."""
    try:
        payload = json.loads(stdin.read())
    except (ValueError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return None
    path = Path(file_path)
    return path if path.is_file() else None


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


def _check(source: Path, stdout: TextIO) -> None:
    # Deferred imports: the rule engine is loaded only for a file that imports JAX.
    from jaxlint.core.config import load_config
    from jaxlint.core.rules import RULE_FOLLOWUPS
    from jaxlint.core.runner import run_checks
    from jaxlint.core.diagnostic import Severity

    cfg = load_config(source.resolve().parent)
    diags = run_checks([source], cfg=cfg, jobs=1, cache_dir=cfg.cache_dir)

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    identities = [_identity(d.rule_id, d.message, _line_text(lines, d.line)) for d in diags]

    state_file = _state_path(cfg.cache_dir, source) if cfg.cache_dir is not None else None
    previous = _load_previous(state_file)
    remaining = previous.copy()
    new: list[Any] = []
    for d, ident in zip(diags, identities):
        if remaining[ident] > 0:
            remaining[ident] -= 1
        else:
            new.append(d)

    if state_file is not None:
        try:
            _store_current(state_file, identities, source)
        except OSError:
            pass  # state is an optimisation; findings are still reported

    if not new:
        return
    text = _format_text(source, new, RULE_FOLLOWUPS)
    if cfg.hook_block_on_error and any(d.severity == Severity.ERROR for d in new):
        payload: dict[str, Any] = {"decision": "block", "reason": text}
    else:
        payload = {
            "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text},
        }
    stdout.write(json.dumps(payload) + "\n")


def _run(stdin: TextIO, stdout: TextIO) -> None:
    source = _read_target(stdin)
    if source is None:
        return
    try:
        raw = source.read_bytes()
    except OSError:
        return
    if not _JAX_IMPORT.search(raw.decode("utf-8", errors="replace")):
        return
    _check(source, stdout)


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
