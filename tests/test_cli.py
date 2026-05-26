import json
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
def test_cli_check_jobs_flag_serial(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["check", "--no-doc", "-j", "1", str(clean_fixture)])
    assert r.exit_code == 0


@pytest.mark.smoke
def test_cli_check_jobs_invalid_exits_2(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["check", "--no-doc", "-j", "0", str(clean_fixture)])
    assert r.exit_code == 2
    assert "jobs must be >= 1" in r.output


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


@pytest.mark.smoke
def test_cli_init_snippet() -> None:
    r = runner.invoke(app, ["init"])
    assert r.exit_code == 0
    assert "[jaxlint]" in r.output
    assert "docstring-style" in r.output


@pytest.mark.integration
def test_cli_check_json_format(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl001_scatter_gather.py"
    r = runner.invoke(app, ["check", "--no-doc", "--format", "json", str(p)])
    assert r.exit_code == 1
    payload = json.loads(r.output)
    assert isinstance(payload, list)
    assert any(row.get("rule_id") == "JL001" for row in payload)


@pytest.mark.integration
def test_cli_doc_json_format(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["doc", "--format", "json", str(clean_fixture)])
    assert r.exit_code == 0
    payload = json.loads(r.output)
    assert payload == []


@pytest.mark.integration
def test_cli_check_github_format(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl001_scatter_gather.py"
    r = runner.invoke(app, ["check", "--no-doc", "--format", "github", str(p)])
    assert r.exit_code == 1
    payload = json.loads(r.output)
    assert "annotations" in payload
    assert any(a.get("title") == "JL001" for a in payload["annotations"])
    assert all("annotation_level" in a for a in payload["annotations"])


@pytest.mark.integration
def test_cli_check_unknown_format_exits_2(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["check", "--no-doc", "--format", "bogusfmt", str(clean_fixture)])
    assert r.exit_code == 2
    assert "Unknown --format" in r.output


@pytest.mark.integration
def test_cli_check_compact_format_on_clean(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["check", "--no-doc", "--format", "compact", str(clean_fixture)])
    assert r.exit_code == 0
    assert not r.output.strip()


@pytest.mark.integration
def test_cli_check_rich_format_on_clean(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["check", "--no-doc", "--format", "rich", str(clean_fixture)])
    assert r.exit_code == 0
    assert "No diagnostics" in r.output


@pytest.mark.integration
def test_cli_check_rich_format_with_violation(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl001_scatter_gather.py"
    r = runner.invoke(app, ["check", "--no-doc", "--format", "rich", str(p)])
    assert r.exit_code == 1
    assert "JL001" in r.output
    assert "Severity" in r.output and "Message" in r.output


@pytest.mark.integration
def test_cli_doc_rich_format_on_clean(clean_fixture: Path) -> None:
    r = runner.invoke(app, ["doc", "--format", "rich", str(clean_fixture)])
    assert r.exit_code == 0
    assert "No diagnostics" in r.output


@pytest.mark.integration
def test_cli_check_sarif_format(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl001_scatter_gather.py"
    r = runner.invoke(app, ["check", "--no-doc", "--format", "sarif", str(p)])
    assert r.exit_code == 1
    payload = json.loads(r.output)
    assert payload.get("version") == "2.1.0"
    results = payload["runs"][0]["results"]
    assert any(r.get("ruleId") == "JL001" for r in results)


@pytest.mark.integration
def test_cli_score_jsonl_and_baseline(tmp_path: Path) -> None:
    current = tmp_path / "current.jsonl"
    baseline_file = tmp_path / "baseline.jsonl"
    entry = {
        "name": "fn",
        "signature": "",
        "docstring": (
            "Brief summary.\n\n"
            "Parameters\n----------\n"
            "x : int\n"
            "    Input value.\n\n"
            "Returns\n-------\n"
            "int\n"
            "    Incremented input.\n"
        ),
    }
    blob = json.dumps(entry) + "\n"
    current.write_text(blob, encoding="utf-8")
    baseline_file.write_text(blob, encoding="utf-8")
    r = runner.invoke(app, ["score", "--threshold", "0.1", "--jsonl", str(current), "--baseline", str(baseline_file)])
    assert r.exit_code == 0
    assert "Mean score:" in r.output
    assert "Baseline mean:" in r.output


@pytest.mark.integration
def test_cli_score_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.py"
    bad.write_text("def oops(\n", encoding="utf-8")
    r = runner.invoke(app, ["score", "--threshold", "0.99", str(bad)])
    assert r.exit_code == 1
    assert "syntax error" in r.output


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


@pytest.mark.smoke
def test_cli_export_default_options() -> None:
    r = runner.invoke(app, ["export"])
    assert r.exit_code == 0
    assert "claude_code" in r.output
    assert "user" in r.output


@pytest.mark.smoke
def test_cli_export_with_surface_and_level() -> None:
    r = runner.invoke(app, ["export", "--surface", "gemini", "--level", "workspace"])
    assert r.exit_code == 0
    assert "gemini" in r.output
    assert "workspace" in r.output
