from pathlib import Path

import pytest
from typer.testing import CliRunner

from jaxlint.cli.main import app

runner = CliRunner()


@pytest.mark.smoke
def test_cli_rules_exit() -> None:
    r = runner.invoke(app, ["rules"])
    assert r.exit_code == 0
    assert "JL001" in r.output


@pytest.mark.smoke
def test_cli_check_clean(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["check", "--no-doc", str(clean_fixture)])
    assert r.exit_code == 0


@pytest.mark.integration
def test_cli_check_with_perf_on_jl001(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl001_scatter_gather.py"
    r = runner.invoke(app, ["check", "--no-doc", str(p)])
    assert r.exit_code == 1  # JL001 is error
    assert "JL001" in r.output


@pytest.mark.integration
def test_cli_doc_on_clean(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["doc", str(clean_fixture)])
    assert r.exit_code == 0


@pytest.mark.integration
def test_cli_score_on_clean(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["score", "--threshold", "0.4", str(clean_fixture)])
    assert r.exit_code == 0


@pytest.mark.smoke
def test_cli_version() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0


@pytest.mark.integration
def test_cli_hlo_when_env_has_no_jax() -> None:
    try:
        import jax  # noqa: F401
    except ImportError:
        jax_present = False
    else:
        jax_present = True

    if jax_present:
        pytest.skip("JAX installed; run needs_jax test for successful hlo path")

    r = runner.invoke(app, ["hlo"])
    assert r.exit_code == 2


@pytest.mark.integration
@pytest.mark.needs_jax
def test_cli_hlo_stub_with_hlo_extra() -> None:
    pytest.importorskip("jax")
    r = runner.invoke(app, ["hlo"])
    assert r.exit_code == 0
    assert "GraphAuditSensor" in r.output
