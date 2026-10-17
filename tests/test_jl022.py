"""JL022: jit-family wrap of a fresh callable inside a host loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.diagnostic import Severity
from jaxlint.core.rules import RULE_FOLLOWUPS, RULES
from jaxlint.core.runner import run_checks
from jaxlint.core.sensors.fresh_jit import fresh_jit_diagnostics
from jaxlint.core.sensors._jax_names import alias_map

import ast


def _jl022(path: Path) -> list:
    diags = run_checks(
        [path], cfg=LintConfig(select=frozenset(["JL"])), perf=True, doc=False, mathjx=False
    )
    return [d for d in diags if d.rule_id == "JL022"]


def _expected(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    prefix = "# jaxlint-expect:"
    assert first.startswith(prefix), f"{path.name} lacks a jaxlint-expect header"
    return first[len(prefix) :].strip()


@pytest.mark.unit
def test_rule_registered_as_warning() -> None:
    entries = {rid: (sev, desc) for rid, sev, desc in RULES}
    assert entries["JL022"][0] is Severity.WARNING
    assert "JL022" in RULE_FOLLOWUPS
    assert "xtrax profile" in RULE_FOLLOWUPS["JL022"]


@pytest.mark.integration
def test_fresh_jit_fixture_flags_every_wrap(fixtures_dir: Path) -> None:
    path = fixtures_dir / "jl022_fresh_jit_in_loop.py"
    assert _expected(path) == "JL022"
    lines = sorted(d.line for d in _jl022(path))
    # lambda, partial, nested def, decorated def, eqx.filter_jit lambda
    assert len(lines) == 5
    assert all(d.severity is Severity.WARNING for d in _jl022(path))
    assert all("retraces each time; hoist the wrap" in d.message for d in _jl022(path))


@pytest.mark.integration
def test_no_flag_fixture_is_silent(fixtures_dir: Path) -> None:
    path = fixtures_dir / "jl022_no_flag.py"
    assert _expected(path) == "none"
    assert _jl022(path) == []


@pytest.mark.unit
def test_sensor_direct_on_inline_source(tmp_path: Path) -> None:
    src = (
        "import jax\n"
        "def f(ys):\n"
        "    for j in range(2):\n"
        "        g = jax.jit(lambda x: x * j)\n"
        "        ys = g(ys)\n"
        "    return ys\n"
    )
    path = tmp_path / "m.py"
    path.write_text(src, encoding="utf-8")
    tree = ast.parse(src)
    assert alias_map(tree)["jax"] == "jax"
    assert fresh_jit_diagnostics(tree, path) == []


@pytest.mark.unit
def test_sensor_flags_constant_lambda_inline(tmp_path: Path) -> None:
    src = (
        "import jax\n"
        "def f(ys):\n"
        "    for _ in range(2):\n"
        "        ys = jax.jit(lambda x: x * 2)(ys)\n"
        "    return ys\n"
    )
    path = tmp_path / "m.py"
    tree = ast.parse(src)
    diags = fresh_jit_diagnostics(tree, path)
    assert [d.rule_id for d in diags] == ["JL022"]
    assert diags[0].line == 4
