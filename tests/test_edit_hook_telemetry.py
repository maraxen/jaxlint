"""``jaxlint-hook`` cisternal spans behind ``CISTERNAL_TELEMETRY``: off means no import, on means
spans without raw paths, and any telemetry failure leaves the hook's output and exit code alone."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import types
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from jaxlint.hooks import edit_hook

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = "jl022_fresh_jit_in_loop.py"


def _project(tmp_path: Path) -> Path:
    # No cache-dir: every run reports every finding, so baseline and telemetry runs compare equal.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'proj'\n")
    return tmp_path


def _target(tmp_path: Path) -> Path:
    project = _project(tmp_path)
    dest = project / "src" / FIXTURE
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / FIXTURE, dest)
    return dest


def _payload(path: Path) -> str:
    return json.dumps({"tool_name": "Edit", "tool_input": {"file_path": str(path)}})


def _run(monkeypatch: pytest.MonkeyPatch, payload: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    code = edit_hook.main()
    return code, out.getvalue()


class _Recorder(AbstractContextManager):
    """A fake cisternal span: records (name, fields) when it exits, like the real end event."""

    events: list[tuple[str, dict[str, Any]]] = []
    raise_on_enter: type[BaseException] | None = None

    def __init__(self, name: str, **fields: Any) -> None:
        self._name = name
        self._fields = fields

    def __enter__(self) -> None:
        if _Recorder.raise_on_enter is not None:
            raise _Recorder.raise_on_enter("span enter failed")

    def __exit__(self, et: Any, ev: Any, tb: Any) -> None:
        _Recorder.events.append((self._name, self._fields))


def _install_fake_cisternal(monkeypatch: pytest.MonkeyPatch) -> None:
    _Recorder.events = []
    _Recorder.raise_on_enter = None
    root = types.ModuleType("cisternal")
    root.get_pipeline = lambda: object()  # type: ignore[attr-defined]
    telemetry = types.ModuleType("cisternal.telemetry")
    span_mod = types.ModuleType("cisternal.telemetry.span")
    span_mod.span = _Recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cisternal", root)
    monkeypatch.setitem(sys.modules, "cisternal.telemetry", telemetry)
    monkeypatch.setitem(sys.modules, "cisternal.telemetry.span", span_mod)


def test_unset_env_never_imports_cisternal(tmp_path: Path) -> None:
    target = _target(tmp_path)
    probe = (
        "import json, sys\n"
        "from jaxlint.hooks import edit_hook\n"
        "code = edit_hook.main()\n"
        "leaked = sorted(m for m in sys.modules if m == 'cisternal' or m.startswith('cisternal.'))\n"
        "assert not leaked, leaked\n"
        "assert code == 0\n"
    )
    import os

    env = {k: v for k, v in os.environ.items() if k != "CISTERNAL_TELEMETRY"}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        input=_payload(target),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "JL022" in result.stdout


def test_enabled_emits_outer_and_check_spans_without_raw_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path)
    monkeypatch.setenv("CISTERNAL_TELEMETRY", "1")
    _install_fake_cisternal(monkeypatch)
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0
    assert "JL022" in json.loads(out)["hookSpecificOutput"]["additionalContext"]

    names = [name for name, _ in _Recorder.events]
    assert names == ["jaxlint.hook.check", "jaxlint.hook"]
    check_fields = _Recorder.events[0][1]
    outer = _Recorder.events[1][1]
    digest = hashlib.sha256(str(target.resolve()).encode("utf-8")).hexdigest()
    assert check_fields == {"path_sha256": digest}
    assert outer["path_sha256"] == digest
    assert outer["tool_name"] == "Edit"
    assert outer["jax_file"] is True
    assert outer["n_findings"] >= 1
    assert outer["n_new"] >= 1
    assert outer["blocked"] is False
    assert isinstance(outer["elapsed_ms"], float)
    assert set(outer) == {
        "tool_name",
        "path_sha256",
        "jax_file",
        "n_findings",
        "n_new",
        "blocked",
        "elapsed_ms",
    }
    raw = str(target.resolve())
    assert all(raw not in str(v) for _, f in _Recorder.events for v in f.values())


def test_non_jax_file_span_reports_jax_file_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    plain = project / "plain.py"
    plain.write_text("def f(x):\n    return x + 1\n")
    monkeypatch.setenv("CISTERNAL_TELEMETRY", "all")
    _install_fake_cisternal(monkeypatch)
    assert _run(monkeypatch, _payload(plain)) == (0, "")
    assert [name for name, _ in _Recorder.events] == ["jaxlint.hook"]
    assert _Recorder.events[0][1]["jax_file"] is False
    assert _Recorder.events[0][1]["n_findings"] == 0


def test_span_failure_leaves_output_and_exit_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    target = _target(tmp_path)
    monkeypatch.delenv("CISTERNAL_TELEMETRY", raising=False)
    baseline = _run(monkeypatch, _payload(target))
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("CISTERNAL_TELEMETRY", "1")
    _install_fake_cisternal(monkeypatch)
    _Recorder.raise_on_enter = RuntimeError
    assert _run(monkeypatch, _payload(target)) == baseline
    assert baseline[0] == 0
    assert capsys.readouterr().err == "jaxlint-hook: telemetry disabled: RuntimeError\n"


def test_cisternal_not_importable_is_silent_and_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    target = _target(tmp_path)
    monkeypatch.delenv("CISTERNAL_TELEMETRY", raising=False)
    baseline = _run(monkeypatch, _payload(target))

    monkeypatch.setenv("CISTERNAL_TELEMETRY", "1")
    monkeypatch.setitem(sys.modules, "cisternal", None)
    monkeypatch.setitem(sys.modules, "cisternal.telemetry.span", None)
    assert _run(monkeypatch, _payload(target)) == baseline
    assert baseline[0] == 0
    assert capsys.readouterr().err == ""


def test_rule_error_still_reaches_internal_error_line_with_spans_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import jaxlint.core.runner as runner

    target = _target(tmp_path)

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "run_checks", boom)
    monkeypatch.setenv("CISTERNAL_TELEMETRY", "1")
    _install_fake_cisternal(monkeypatch)
    code, out = _run(monkeypatch, _payload(target))
    assert (code, out) == (0, "")
    assert capsys.readouterr().err == "jaxlint-hook: internal error: RuntimeError: boom\n"
    # The check span closed on the body's exception, and the outer span still closed after it.
    assert [name for name, _ in _Recorder.events] == ["jaxlint.hook.check", "jaxlint.hook"]
