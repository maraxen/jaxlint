"""JL020: host loop dispatches a jitted function or jnp op per element and collects results."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.diagnostic import Severity
from jaxlint.core.rules import KNOWN_RULE_IDS, RULE_FOLLOWUPS
from jaxlint.core.runner import run_checks
from jaxlint.core.sensors.host_loop import host_loop_diagnostics


def _diags(src: str, *, jnp_ops: bool = False) -> list:
    return host_loop_diagnostics(ast.parse(src), Path("m.py"), jnp_ops=jnp_ops)


def _header(path: Path) -> str:
    return path.read_text(encoding="utf-8").splitlines()[0]


def _hit_lines(path: Path) -> set[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {i for i, line in enumerate(lines, start=1) if line.rstrip().endswith("# hit")}


def _config_from_header(path: Path) -> LintConfig:
    lines = path.read_text(encoding="utf-8").splitlines()
    jnp_ops = any(line.strip() == "# jaxlint-config: host-loop-jnp-ops=true" for line in lines)
    return LintConfig(select=frozenset(["JL020"]), host_loop_jnp_ops=jnp_ops)


@pytest.mark.integration
def test_fixture_host_loop_flags_marked_loops(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl020_host_loop_jitted.py"
    assert "JL020" in _header(p)  # header lists every rule the fixture triggers
    diags = run_checks([p], cfg=_config_from_header(p), perf=True, doc=False)
    assert {d.rule_id for d in diags} == {"JL020"}
    assert {d.line for d in diags} == _hit_lines(p)
    assert all(d.severity == Severity.INFO for d in diags)


@pytest.mark.integration
def test_fixture_jnp_loop_flags_only_when_enabled(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl020_host_loop_jnp.py"
    assert "JL020" in _header(p)  # header lists every rule the fixture triggers
    assert "# jaxlint-config: host-loop-jnp-ops=true" in p.read_text(encoding="utf-8")
    diags = run_checks([p], cfg=_config_from_header(p), perf=True, doc=False)
    assert {d.line for d in diags} == _hit_lines(p)
    off = run_checks([p], cfg=LintConfig(select=frozenset(["JL020"])), perf=True, doc=False)
    assert off == []


@pytest.mark.integration
def test_fixture_no_flag_is_silent(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl020_no_flag.py"
    assert "JL020" not in _header(p)
    diags = run_checks([p], cfg=LintConfig(select=frozenset(["JL020"])), perf=True, doc=False)
    assert diags == []


@pytest.mark.unit
def test_local_jit_append_flagged_with_message() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        out.append(f(x))\n"
        "    return out\n"
    )
    diags = _diags(src)
    assert [(d.rule_id, d.line) for d in diags] == [("JL020", 7)]
    assert diags[0].message == (
        "Loop calls `f` once per element and collects results; dispatch is per element — "
        "batch with jax.vmap over the loop axis."
    )


@pytest.mark.unit
def test_jnp_from_import_alias_flagged_when_enabled() -> None:
    src = (
        "from jax import numpy as jnp\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        out.append(jnp.sum(x))\n"
        "    return out\n"
    )
    assert [d.line for d in _diags(src, jnp_ops=True)] == [4]
    assert _diags(src) == []


@pytest.mark.unit
def test_jnp_default_off_even_with_jnp_names() -> None:
    src = (
        "import jax.numpy as jnp\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        out.append(jnp.asarray(x))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_name_assigned_from_call_then_appended_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        y = f(x)\n        out.append(y)\n"
        "    return out\n"
    )
    assert [d.line for d in _diags(src)] == [7]


@pytest.mark.unit
def test_subscript_assignment_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs, arr):\n"
        "    for i, x in enumerate(xs):\n        arr[i] = f(x)\n"
        "    return arr\n"
    )
    assert [d.line for d in _diags(src)] == [6]


@pytest.mark.unit
def test_loop_inside_kernel_not_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "@jax.jit\n"
        "def k(xs):\n    out = []\n"
        "    for x in xs:\n        out.append(f(x))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_carry_recurrence_not_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def step(x):\n    return x\n"
        "def g(x, n):\n    out = []\n"
        "    for _ in range(n):\n        x = step(x)\n        out.append(x)\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_derived_carry_not_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def step(x):\n    return x\n"
        "def g(x, n):\n    out = []\n"
        "    for _ in range(n):\n        y = step(x)\n        x = y + 1\n        out.append(y)\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_timing_loop_not_flagged() -> None:
    src = (
        "import time\nimport jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        t0 = time.perf_counter()\n"
        "        out.append(f(x))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_block_until_ready_loop_not_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        out.append(f(x).block_until_ready())\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_imported_name_ending_in_jit_not_flagged() -> None:
    src = (
        "from elsewhere import step_jit\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n        out.append(step_jit(x))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_call_result_not_collected_not_flagged() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs):\n    for x in xs:\n        f(x)\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_nested_loops_report_inner_loop_only() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xss):\n    out = []\n"
        "    for xs in xss:\n"
        "        for x in xs:\n"
        "            out.append(f(x))\n"
        "    return out\n"
    )
    assert [d.line for d in _diags(src)] == [8]


@pytest.mark.unit
def test_rule_registered_with_followup() -> None:
    assert "JL020" in KNOWN_RULE_IDS
    followup = RULE_FOLLOWUPS["JL020"]
    assert "xtrax profile" in followup
    assert "--scale-axis" in followup
    assert "jax.vmap" in followup


@pytest.mark.unit
def test_serial_recurrence_through_other_call_skips_loop() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "@jax.jit\n"
        "def apply_j(s):\n    return s\n"
        "def g(state, n):\n    out = []\n"
        "    for _ in range(n):\n        state = apply_j(state)\n"
        "        out.append(f(state))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_tuple_carry_through_call_skips_loop() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(x, k, n):\n    out = []\n"
        "    for _ in range(n):\n        x, k = step2(x, k)\n"
        "        out.append(f(x))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["time_energy", "run_timed", "timeit_once", "bench_fn", "profile_step", "perf_counter_step"],
)
def test_timing_segment_helper_skips_loop(name: str) -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        f"def g(xs):\n    out = []\n"
        "    for x in xs:\n"
        f"        out.append({name}(f, x))\n"
        "    return out\n"
    )
    assert _diags(src) == []


@pytest.mark.unit
def test_timestep_name_is_not_a_timing_guard() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(x):\n    return x\n"
        "def g(xs):\n    out = []\n"
        "    for x in xs:\n"
        "        out.append(f(timestep(x)))\n"
        "    return out\n"
    )
    assert [d.line for d in _diags(src)] == [7]


@pytest.mark.unit
def test_host_loop_jnp_ops_config_key_parses(tmp_path: Path) -> None:
    from jaxlint.core.config import load_config

    (tmp_path / "pyproject.toml").write_text(
        "[tool.jaxlint]\nhost-loop-jnp-ops = true\n", encoding="utf-8"
    )
    assert load_config(tmp_path).host_loop_jnp_ops is True
    assert LintConfig().host_loop_jnp_ops is False


@pytest.mark.unit
def test_host_loop_jnp_ops_is_in_cache_namespace() -> None:
    from jaxlint.core.cache import _canonical_config_dict

    on = _canonical_config_dict(LintConfig(host_loop_jnp_ops=True))
    off = _canonical_config_dict(LintConfig(host_loop_jnp_ops=False))
    assert on["host_loop_jnp_ops"] is True and off["host_loop_jnp_ops"] is False
