"""``jaxlint-hook`` PostToolUse behaviour: new-finding feedback, dedupe, blocking, failure policy."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jaxlint.hooks import edit_hook

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _project(tmp_path: Path, *, cache: bool = True, block: bool | None = None) -> Path:
    """A throwaway project whose config is found by the walk from the file's directory."""
    lines = ["[tool.jaxlint]"]
    if cache:
        lines.append('cache-dir = "cache"')
    if block is not None:
        lines.append(f"hook-block-on-error = {'true' if block else 'false'}")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'proj'\n" + "\n".join(lines) + "\n")
    return tmp_path


def _copy_fixture(project: Path, name: str) -> Path:
    dest = project / "src" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / name, dest)
    return dest


def _run(monkeypatch: pytest.MonkeyPatch, payload: str) -> tuple[int, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    code = edit_hook.main()
    return code, out.getvalue()


def _payload(path: Path, tool: str = "Edit") -> str:
    return json.dumps({"tool_name": tool, "tool_input": {"file_path": str(path)}})


def _context(stdout: str) -> str:
    return json.loads(stdout)["hookSpecificOutput"]["additionalContext"]


def test_non_jax_file_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    target = project / "plain.py"
    target.write_text("def f(x):\n    return x + 1\n")
    code, out = _run(monkeypatch, _payload(target, "Write"))
    assert code == 0
    assert out == ""


def test_non_python_file_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    target = project / "notes.md"
    target.write_text("import jax\n")
    code, out = _run(monkeypatch, _payload(target))
    assert (code, out) == (0, "")


def test_new_finding_reported_with_followup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    target = _copy_fixture(
        project, "jl022_fresh_jit_in_loop.py"
    )  # JL022 has a RULE_FOLLOWUPS entry
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0
    text = _context(out)
    assert "JL022 warning:" in text
    assert f"{target}:" in text
    assert "  → Confirm with `xtrax profile" in text


def test_unchanged_file_is_deduped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    first_code, first = _run(monkeypatch, _payload(target))
    assert first_code == 0 and "JL003" in _context(first)
    second_code, second = _run(monkeypatch, _payload(target))
    assert (second_code, second) == (0, "")


def test_line_shift_alone_is_not_new(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    _run(monkeypatch, _payload(target))
    target.write_text("\n" + target.read_text())
    code, out = _run(monkeypatch, _payload(target))
    assert (code, out) == (0, "")


def test_state_file_honours_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    _run(monkeypatch, _payload(target))
    states = list((project / "cache" / edit_hook.STATE_SUBDIR).glob("*.json"))
    assert len(states) == 1
    state = json.loads(states[0].read_text())
    assert state["path"] == str(target.resolve())
    assert {rid for rid, _msg, _h in state["identities"]} >= {"JL003"}


def test_no_cache_dir_means_every_finding_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, cache=False)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    _run(monkeypatch, _payload(target))
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0 and "JL003" in _context(out)
    assert not (project / "cache").exists()


def _edit_payload(path: Path, new_string: str) -> str:
    return json.dumps(
        {
            "tool_name": "Edit",
            "tool_input": {"file_path": str(path), "old_string": "x", "new_string": new_string},
        }
    )


def test_no_cache_dir_edit_reports_only_written_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, cache=False)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    _, full = _run(monkeypatch, _payload(target, "Write"))
    first = _context(full).splitlines()[0].removeprefix(f"{target}:")
    line_no, rule_id = int(first.split()[0]), first.split()[1]
    flagged = target.read_text().splitlines()[line_no - 1]

    code, out = _run(monkeypatch, _edit_payload(target, flagged + "\n"))
    assert code == 0 and rule_id in _context(out)
    assert all(f"{target}:{line_no} " in ln for ln in _context(out).splitlines() if ln[0] != " ")


def test_no_cache_dir_edit_elsewhere_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, cache=False)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    code, out = _run(monkeypatch, _edit_payload(target, "# an unrelated comment\n"))
    assert (code, out) == (0, "")


def test_garbage_stdin_exits_zero_silently(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    code, out = _run(monkeypatch, "this is not json {")
    assert (code, out) == (0, "")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "payload", ["[]", "{}", '{"tool_input": "x"}', '{"tool_input": {"file_path": 3}}']
)
def test_malformed_payload_shapes_exit_zero(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    assert _run(monkeypatch, payload) == (0, "")


def test_missing_file_is_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out = _run(monkeypatch, _payload(tmp_path / "gone.py"))
    assert (code, out) == (0, "")


def test_internal_error_exits_zero_with_stderr_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import jaxlint.core.runner as runner

    project = _project(tmp_path)
    target = _copy_fixture(project, "jl003_loop_kernel.py")

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "run_checks", boom)
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0
    assert out == ""
    err = capsys.readouterr().err
    assert err == "jaxlint-hook: internal error: RuntimeError: boom\n"


def test_block_flag_on_with_error_finding_emits_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, block=True)
    target = _copy_fixture(project, "jl003_loop_kernel.py")  # JL003 is error severity
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0
    payload = json.loads(out)
    assert payload["decision"] == "block"
    assert "JL003" in payload["reason"]
    assert "hookSpecificOutput" not in payload


def test_block_flag_on_with_only_warnings_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, block=True)
    target = _copy_fixture(project, "jl002_short_jit.py")  # JL002 is a warning, no error finding
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0
    assert "decision" not in json.loads(out)
    assert "JL002" in _context(out)


def test_block_flag_off_emits_context_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, block=False)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    code, out = _run(monkeypatch, _payload(target))
    assert code == 0
    payload = json.loads(out)
    assert set(payload) == {"hookSpecificOutput"}
    assert "JL003" in payload["hookSpecificOutput"]["additionalContext"]


def test_module_entry_point_runs_as_script(tmp_path: Path) -> None:
    project = _project(tmp_path)
    target = _copy_fixture(project, "jl003_loop_kernel.py")
    payload = _payload(target)
    proc = subprocess.run(
        [sys.executable, "-m", "jaxlint.hooks.edit_hook"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "JL003" in _context(proc.stdout)
