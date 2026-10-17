"""JL023 per-iteration host sync on a device value (info, src/ by default)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig, load_config
from jaxlint.core.runner import run_checks

JL_ONLY = frozenset(["JL023"])


def _expect(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    m = re.fullmatch(r"# jaxlint-expect: (JL023|none)", first)
    assert m, f"{path}: first line must be '# jaxlint-expect: JL023' or '... none'"
    return m.group(1)


def _jl023_lines(path: Path, cfg: LintConfig) -> set[int]:
    diags = run_checks([path], cfg=cfg, perf=True, doc=False, mathjx=False)
    return {d.line for d in diags if d.rule_id == "JL023"}


def test_positive_fixture_flags_exactly_the_marked_lines(fixtures_dir: Path) -> None:
    p = fixtures_dir / "src" / "jl023_per_iteration_sync.py"
    assert _expect(p) == "JL023"
    marked = {
        i
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1)
        if "# sync-here" in line
    }
    got = _jl023_lines(p, LintConfig(select=JL_ONLY))
    assert got == marked


def test_positive_fixture_messages_name_the_op_and_value(fixtures_dir: Path) -> None:
    p = fixtures_dir / "src" / "jl023_per_iteration_sync.py"
    diags = run_checks([p], cfg=LintConfig(select=JL_ONLY), perf=True, doc=False, mathjx=False)
    msgs = [d.message for d in diags if d.rule_id == "JL023"]
    assert msgs, "expected JL023 diagnostics"
    pattern = re.compile(
        r"^`.+` on device value `.+` inside a loop forces a host sync every iteration; "
        r"keep values on device across the loop and transfer once\.$"
    )
    assert all(pattern.match(m) for m in msgs), msgs
    assert all(d.severity.value == "info" for d in diags if d.rule_id == "JL023")


@pytest.mark.parametrize("name", ["jl023_no_flag.py"])
def test_negative_fixture_is_silent(fixtures_dir: Path, name: str) -> None:
    p = fixtures_dir / "src" / name
    assert _expect(p) == "none"
    assert _jl023_lines(p, LintConfig(select=JL_ONLY)) == set()


def test_default_scope_skips_files_outside_src(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl023_outside_src_no_flag.py"
    assert _expect(p) == "none"
    assert _jl023_lines(p, LintConfig(select=JL_ONLY)) == set()


def test_host_sync_everywhere_fires_outside_src(fixtures_dir: Path) -> None:
    p = fixtures_dir / "jl023_outside_src_no_flag.py"
    cfg = LintConfig(select=JL_ONLY, host_sync_everywhere=True)
    assert _jl023_lines(p, cfg) == {12}


def test_host_sync_everywhere_config_key_parses(tmp_path: Path) -> None:
    cfg_path = tmp_path / "jaxlint.toml"
    cfg_path.write_text("[jaxlint]\nhost-sync-everywhere = true\n", encoding="utf-8")
    assert load_config(cfg_path).host_sync_everywhere is True


def test_host_sync_everywhere_defaults_false(tmp_path: Path) -> None:
    cfg_path = tmp_path / "jaxlint.toml"
    cfg_path.write_text("[jaxlint]\nselect = []\n", encoding="utf-8")
    assert load_config(cfg_path).host_sync_everywhere is False


def test_jl023_is_registered_as_info_with_followup() -> None:
    from jaxlint.core.diagnostic import Severity
    from jaxlint.core.rules import RULE_FOLLOWUPS, RULES

    entries = {rid: sev for rid, sev, _ in RULES}
    assert entries["JL023"] is Severity.INFO
    assert "xtrax profile" in RULE_FOLLOWUPS["JL023"]


def test_each_sync_reported_once() -> None:
    """Module scope must not walk into function bodies (each site reported exactly once)."""

    fixture = Path(__file__).parent / "fixtures" / "src" / "jl023_per_iteration_sync.py"
    markers = fixture.read_text(encoding="utf-8").count("# sync-here")
    diags = run_checks([fixture], cfg=LintConfig(), perf=True, doc=False, mathjx=False)
    hits = [d for d in diags if d.rule_id == "JL023"]
    assert len(hits) == markers
    assert len({(d.line, d.col) for d in hits}) == markers
