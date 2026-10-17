"""JL021: host loop over replicates/resamples should batch the replicate axis."""

from __future__ import annotations

import ast
from pathlib import Path


from jaxlint.core.config import LintConfig
from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.runner import run_checks
from jaxlint.core.rules import RULE_FOLLOWUPS, RULES
from jaxlint.core.sensors import perf_diagnostics_from_tree


def _jl021(src: str) -> list[Diagnostic]:
    tree = ast.parse(src)
    diags = perf_diagnostics_from_tree(Path("snippet.py"), LintConfig(), tree)
    return [d for d in diags if d.rule_id == "JL021"]


def _lines(src: str) -> list[int]:
    return [d.line for d in _jl021(src)]


# --- positives -------------------------------------------------------------------


def test_numpy_bootstrap_append_flags_outer_for_line() -> None:
    src = (
        "import numpy as np\n"
        "def f(data, n_boot):\n"
        "    out = []\n"
        "    for _ in range(n_boot):\n"
        "        s = np.random.choice(data, size=len(data))\n"
        "        out.append(np.mean(s))\n"
        "    return out\n"
    )
    diags = _jl021(src)
    assert [d.line for d in diags] == [4]
    assert diags[0].severity == Severity.INFO
    assert "Host loop over n_boot replicates" in diags[0].message
    assert "np.random.choice" in diags[0].message
    assert "xtrax profile" in diags[0].message


def test_subscript_assignment_over_name_iterable_flags() -> None:
    src = (
        "def f(boot_keys, fit):\n"
        "    res = {}\n"
        "    for key in boot_keys:\n"
        "        res[key] = fit(key)\n"
        "    return res\n"
    )
    assert _lines(src) == [3]


def test_enumerate_over_replicate_name_with_augassign_flags() -> None:
    src = (
        "def f(boot_samples, g):\n"
        "    total = 0.0\n"
        "    for i, s in enumerate(boot_samples):\n"
        "        total += g(s)\n"
        "    return total\n"
    )
    assert _lines(src) == [3]


def test_range_with_start_and_replicate_stop_flags() -> None:
    src = (
        "def f(x, n_replicates, fit):\n"
        "    arr = [None] * n_replicates\n"
        "    for i in range(0, n_replicates):\n"
        "        arr[i] = fit(x, i)\n"
        "    return arr\n"
    )
    assert _lines(src) == [3]


def test_scipy_per_replicate_fit_subscript_flags() -> None:
    src = (
        "import numpy as np\n"
        "from scipy.optimize import minimize\n"
        "def f(x0, n_boot):\n"
        "    fits = np.zeros((n_boot, 2))\n"
        "    for b in range(n_boot):\n"
        "        res = minimize(lambda p: p @ p, x0)\n"
        "        fits[b] = res.x\n"
        "    return fits\n"
    )
    diags = _jl021(src)
    assert [d.line for d in diags] == [5]
    assert "minimize" in diags[0].message


def test_permutation_and_seed_names_match() -> None:
    src = "def f(n_perm, stat, out):\n    for _ in range(n_perm):\n        out.append(stat())\n"
    assert _lines(src) == [2]


# --- negatives -------------------------------------------------------------------


def test_print_only_body_does_not_flag() -> None:
    src = "def f(n_boot):\n    for b in range(n_boot):\n        print(b)\n"
    assert _lines(src) == []


def test_unrelated_range_with_append_does_not_flag() -> None:
    src = (
        "def f(n, fn):\n"
        "    out = []\n"
        "    for i in range(n):\n"
        "        out.append(fn(i))\n"
        "    return out\n"
    )
    assert _lines(src) == []


def test_replicate_loop_without_accumulation_does_not_flag() -> None:
    src = "def f(x, n_boot, fit):\n    for _ in range(n_boot):\n        fit(x)\n"
    assert _lines(src) == []


def test_mc_replica_loop_is_not_a_resample_axis() -> None:
    src = (
        "def f(n_replicas, run_one):\n"
        "    out = []\n"
        "    for r in range(n_replicas):\n"
        "        out.append(run_one(r))\n"
        "    return out\n"
    )
    assert _lines(src) == []


def test_builtin_scalar_conversion_is_trivial() -> None:
    src = (
        "def f(n_boot, xs):\n"
        "    out = []\n"
        "    for b in range(n_boot):\n"
        "        out.append(float(xs[b]))\n"
        "    return out\n"
    )
    assert _lines(src) == []


def test_replicate_loop_with_only_trivial_calls_does_not_flag() -> None:
    src = (
        "def f(n_boot, xs):\n"
        "    out = []\n"
        "    for b in range(n_boot):\n"
        "        out.append(str(b))\n"
        "        out.append(len(xs))\n"
        "    return out\n"
    )
    assert _lines(src) == []


def test_timing_loop_with_only_time_calls_does_not_flag() -> None:
    src = (
        "import time\n"
        "def f(n_reps):\n"
        "    ts = []\n"
        "    for _ in range(n_reps):\n"
        "        t0 = time.perf_counter()\n"
        "        ts.append(time.perf_counter() - t0)\n"
        "    return ts\n"
    )
    assert _lines(src) == []


def test_loop_inside_jit_kernel_is_skipped() -> None:
    src = (
        "import jax\n"
        "@jax.jit\n"
        "def f(n_boot, x, fit):\n"
        "    acc = []\n"
        "    for _ in range(n_boot):\n"
        "        acc.append(fit(x))\n"
        "    return acc\n"
    )
    assert _lines(src) == []


def test_loop_inside_eqx_grad_kernel_is_skipped() -> None:
    src = (
        "import equinox as eqx\n"
        "@eqx.filter_grad\n"
        "def f(n_boot, x, fit):\n"
        "    acc = []\n"
        "    for _ in range(n_boot):\n"
        "        acc.append(fit(x))\n"
        "    return acc\n"
    )
    assert _lines(src) == []


def test_nested_same_name_loops_report_outer_only() -> None:
    src = (
        "def f(n_boot, xs, fit):\n"
        "    out = []\n"
        "    for _ in range(n_boot):\n"
        "        row = []\n"
        "        for _ in range(n_boot):\n"
        "            row.append(fit(xs))\n"
        "        out.append(row)\n"
        "    return out\n"
    )
    assert _lines(src) == [3]


def test_nested_different_name_loops_are_both_reported() -> None:
    src = (
        "def f(n_boot, n_perm, xs, fit):\n"
        "    out = []\n"
        "    for _ in range(n_boot):\n"
        "        row = []\n"
        "        for _ in range(n_perm):\n"
        "            row.append(fit(xs))\n"
        "        out.append(row)\n"
        "    return out\n"
    )
    assert _lines(src) == [3, 5]


# --- registration / fixtures -----------------------------------------------------


def test_rule_registered_as_info() -> None:
    entries = [r for r in RULES if r[0] == "JL021"]
    assert len(entries) == 1
    assert entries[0][1] == Severity.INFO


def test_rule_followup_points_at_xtrax_profile() -> None:
    text = RULE_FOLLOWUPS["JL021"]
    assert "xtrax profile" in text
    assert "0.5" in text
    assert "jax.vmap" in text


def test_bootstrap_fixture_flags_marked_loops(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl021_bootstrap_loop.py"
    diags = run_checks(
        [p], cfg=LintConfig(select=frozenset(["JL"])), perf=True, doc=False, mathjx=False
    )
    got = sorted(d.line for d in diags if d.rule_id == "JL021")
    expected = _marked_lines(p, "# flag")
    assert got == expected
    assert len(expected) == 2


def test_no_flag_fixture_is_clean(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl021_replicate_no_flag.py"
    diags = run_checks(
        [p], cfg=LintConfig(select=frozenset(["JL"])), perf=True, doc=False, mathjx=False
    )
    assert [d for d in diags if d.rule_id == "JL021"] == []


def test_clean_minimal_has_no_diagnostics(clean_fixture: Path) -> None:
    diags = run_checks([clean_fixture], cfg=LintConfig(), perf=True, doc=True, mathjx=True)
    assert diags == []


def _marked_lines(path: Path, marker: str) -> list[int]:
    return [
        i
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.rstrip().endswith(marker)
    ]
