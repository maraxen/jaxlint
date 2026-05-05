from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.runner import run_checks



@pytest.mark.integration
def test_jl001_fixture(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl001_scatter_gather.py"
    jl_only = LintConfig(select=frozenset(["JL"]))
    diags = run_checks([p], cfg=jl_only, perf=True, doc=False, mathjx=False)
    assert "JL001" in {d.rule_id for d in diags}


@pytest.mark.integration
def test_clean_fixture_no_errors(fixtures_dir: Path) -> None:
    p = fixtures_dir / "clean_minimal.py"
    diags = run_checks([p], cfg=LintConfig(), perf=True, doc=True, mathjx=True)
    errors = [d for d in diags if d.severity.value == "error"]
    assert not errors


@pytest.mark.integration
def test_jm001_unterminated_block(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jm001_bad_math.py"
    diags = run_checks([p], cfg=LintConfig(), perf=False, doc=True, mathjx=True)
    assert "JM001" in {d.rule_id for d in diags}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "rule_id"),
    [
        ("jl002_short_jit.py", "JL002"),
        ("jl003_loop_kernel.py", "JL003"),
        ("jl004_side_effect.py", "JL004"),
        ("jl005_shape_in_kernel.py", "JL005"),
        ("jl010_float64.py", "JL010"),
        ("jl011_sharding_spec.py", "JL011"),
        ("jl012_div_jit.py", "JL012"),
    ],
)
def test_perf_rule_fixtures(fixtures_dir: Path, filename: str, rule_id: str) -> None:
    p = fixtures_dir / filename
    jl_only = LintConfig(select=frozenset(["JL"]))
    diags = run_checks([p], cfg=jl_only, perf=True, doc=False, mathjx=False)
    ids = {d.rule_id for d in diags}
    assert rule_id in ids


@pytest.mark.integration
def test_jl005_strict_shapes_error(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl005_shape_in_kernel.py"
    jl_only = LintConfig(select=frozenset(["JL"]), strict_shapes=True)
    diags = run_checks([p], cfg=jl_only, perf=True, doc=False, mathjx=False)
    jl005 = [d for d in diags if d.rule_id == "JL005"]
    assert jl005 and all(d.severity.value == "error" for d in jl005)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "rule_id"),
    [
        ("jd001_missing_docstring.py", "JD001"),
        ("jd002_missing_parameters.py", "JD002"),
        ("jd004_param_mismatch.py", "JD004"),
        ("jd005_no_returns_section.py", "JD005"),
        ("jm002_bad_latex.py", "JM002"),
    ],
)
def test_doc_math_rule_fixtures(fixtures_dir: Path, filename: str, rule_id: str) -> None:
    p = fixtures_dir / filename
    prefix = rule_id[:2]
    selective = LintConfig(select=frozenset([prefix]))
    diags = run_checks([p], cfg=selective, perf=False, doc=True, mathjx=True)
    assert rule_id in {d.rule_id for d in diags}


@pytest.mark.integration
def test_doc_select_limits_doc_rules(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jm002_bad_latex.py"
    jm_only = LintConfig(select=frozenset(["JM"]))
    diags = run_checks([p], cfg=jm_only, perf=False, doc=True, mathjx=True)
    assert {d.rule_id for d in diags}.issubset({"JM001", "JM002", "JM003"})

@pytest.mark.integration
def test_jm003_large_math_fixture(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jm003_large_math.py"
    diags = run_checks(
        [p], cfg=LintConfig(select=frozenset(["JM"])), perf=False, doc=True, mathjx=True
    )
    assert "JM003" in {d.rule_id for d in diags}


@pytest.mark.unit
def test_rule_enabled_per_file_ignore() -> None:
    """Sanity-check config wiring without running full checks."""
    from jaxlint.core.config import rule_enabled

    cfg = LintConfig(
        select=frozenset(["JL"]),
        ignore=frozenset(),
        per_file_ignores={"tests/fixtures/*.py": ("JL*",)},
    )
    p = Path("tests/fixtures/jl012_div_jit.py")
    assert rule_enabled("JL012", cfg, p) is False

