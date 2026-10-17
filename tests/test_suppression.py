"""Inline suppression comments: ``# jaxlint: ignore[...]`` and bare ``# jaxlint: ignore``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jaxlint.cli.main import app
from jaxlint.core.config import LintConfig
from jaxlint.core.runner import diagnostics_for_python_source, resolve_paths, run_checks

runner = CliRunner()

# Line 7 is the `for` loop (JL003 fires there); line 9 is the division (JL012).
LOOP_TEMPLATE = '''import jax


@jax.jit
def f(x):
    """Doc."""
    for i in range(3):{loop_comment}
        x = x + i
    return x / 2{div_comment}
'''


def _source(loop_comment: str = "", div_comment: str = "") -> str:
    return LOOP_TEMPLATE.format(loop_comment=loop_comment, div_comment=div_comment)


def _rule_ids(source: str, tmp_path: Path) -> set[str]:
    p = tmp_path / "mod.py"
    diags = diagnostics_for_python_source(
        p, source, cfg=LintConfig(), perf=True, doc=False, mathjx=False
    )
    return {d.rule_id for d in diags}


def _lines_for(source: str, rule_id: str, tmp_path: Path) -> list[int]:
    p = tmp_path / "mod.py"
    diags = diagnostics_for_python_source(
        p, source, cfg=LintConfig(), perf=True, doc=False, mathjx=False
    )
    return sorted(d.line for d in diags if d.rule_id == rule_id)


@pytest.mark.unit
def test_baseline_reports_jl003_on_loop_line(tmp_path: Path) -> None:
    assert _lines_for(_source(), "JL003", tmp_path) == [7]


@pytest.mark.unit
def test_single_id_suppresses_that_rule_on_its_line(tmp_path: Path) -> None:
    src = _source(loop_comment="  # jaxlint: ignore[JL003]")
    assert "JL003" not in _rule_ids(src, tmp_path)


@pytest.mark.unit
def test_list_suppresses_each_listed_rule(tmp_path: Path) -> None:
    # One-line loop body: the division and the loop share line 7.
    src = (
        "import jax\n\n\n@jax.jit\ndef f(x):\n"
        '    """Doc."""\n'
        "    for i in range(3): x = x / 2  # jaxlint: ignore[JL003,JL012]\n"
        "    return x\n"
    )
    ids = _rule_ids(src, tmp_path)
    assert "JL003" not in ids
    assert "JL012" not in ids


@pytest.mark.unit
def test_bare_ignore_suppresses_all_rules_on_line(tmp_path: Path) -> None:
    src = (
        "import jax\n\n\n@jax.jit\ndef f(x):\n"
        '    """Doc."""\n'
        "    for i in range(3): x = x / 2  # jaxlint: ignore\n"
        "    return x\n"
    )
    ids = _rule_ids(src, tmp_path)
    assert "JL003" not in ids
    assert "JL012" not in ids


@pytest.mark.unit
def test_wrong_id_leaves_diagnostic(tmp_path: Path) -> None:
    src = _source(loop_comment="  # jaxlint: ignore[JL012]")
    assert _lines_for(src, "JL003", tmp_path) == [7]


@pytest.mark.unit
def test_whitespace_tolerant_parsing(tmp_path: Path) -> None:
    src = _source(loop_comment="  #jaxlint:ignore[ JL003 ,JL012 ]")
    assert "JL003" not in _rule_ids(src, tmp_path)


@pytest.mark.unit
def test_rule_ids_are_case_sensitive(tmp_path: Path) -> None:
    src = _source(loop_comment="  # jaxlint: ignore[jl003]")
    assert _lines_for(src, "JL003", tmp_path) == [7]


@pytest.mark.unit
def test_hash_inside_string_literal_is_not_a_comment(tmp_path: Path) -> None:
    # The loop line contains the marker only inside a string literal.
    src = (
        "import jax\n\n\n@jax.jit\ndef f(x):\n"
        '    """Doc."""\n'
        '    for i in range(len("# jaxlint: ignore")):\n'
        "        x = x + i\n"
        "    return x\n"
    )
    assert _lines_for(src, "JL003", tmp_path) == [7]


@pytest.mark.unit
def test_suppression_on_other_line_does_not_apply(tmp_path: Path) -> None:
    # Comment sits on the `def` line (5), not the loop line (7).
    src = _source().replace("def f(x):", "def f(x):  # jaxlint: ignore[JL003]")
    assert _lines_for(src, "JL003", tmp_path) == [7]


@pytest.mark.unit
def test_unknown_rule_id_is_silently_inert(tmp_path: Path) -> None:
    src = _source(loop_comment="  # jaxlint: ignore[JL999]")
    assert _lines_for(src, "JL003", tmp_path) == [7]


@pytest.mark.integration
def test_runner_api_honours_suppression(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(_source(loop_comment="  # jaxlint: ignore[JL003]"), encoding="utf-8")
    diags = run_checks(
        resolve_paths([p]), cfg=LintConfig(), perf=True, doc=False, mathjx=False, jobs=1
    )
    assert "JL003" not in {d.rule_id for d in diags}


@pytest.mark.integration
def test_cli_path_honours_suppression(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text(_source(loop_comment="  # jaxlint: ignore[JL003]"), encoding="utf-8")
    result = runner.invoke(app, ["check", str(p), "--format", "compact", "--no-doc", "--jobs", "1"])
    assert "JL003" not in result.output


@pytest.mark.integration
def test_cache_cannot_return_pre_suppression_diagnostics(tmp_path: Path) -> None:
    """Editing a comment in place (same size, same mtime) must not serve a stale cache entry."""
    p = tmp_path / "mod.py"
    p.write_text(_source(loop_comment="  # jaxlint: ignore[JL012]"), encoding="utf-8")
    st = p.stat()
    cache = tmp_path / "cache"

    def run() -> set[str]:
        diags = run_checks(
            [p], cfg=LintConfig(), perf=True, doc=False, mathjx=False, jobs=1, cache_dir=cache
        )
        return {d.rule_id for d in diags}

    assert "JL003" in run()

    p.write_text(_source(loop_comment="  # jaxlint: ignore[JL003]"), encoding="utf-8")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert p.stat().st_size == st.st_size
    assert "JL003" not in run()
