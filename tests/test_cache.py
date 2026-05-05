"""Disk cache for ``run_checks`` (opt-in ``cache_dir``)."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from jaxlint.cli.main import app
from jaxlint.core.cache import namespace_segment
from jaxlint.core.config import LintConfig, load_config
from jaxlint.core.diagnostic import Severity
from jaxlint.core.runner import resolve_paths, run_checks

runner = CliRunner()


@pytest.mark.unit
def test_namespace_segment_differs_when_only_severity_overrides_change() -> None:
    """Fingerprint must move when overrides change so cached diagnostics cannot cross namespaces."""
    base = LintConfig()
    tweaked = LintConfig(severity_overrides={"JL001": Severity.WARNING})
    a = namespace_segment(base, perf=True, doc=False, mathjx=False)
    b = namespace_segment(tweaked, perf=True, doc=False, mathjx=False)
    assert a != b


@pytest.mark.integration
def test_cache_warm_matches_cold_serial(fixtures_dir: Path, tmp_path: Path) -> None:
    paths = resolve_paths([fixtures_dir / "clean_minimal.py"])
    cfg = LintConfig()
    cache = tmp_path / "cache"
    cold = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    warm = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    assert cold == warm


@pytest.mark.integration
def test_cache_warm_matches_parallel(fixtures_dir: Path, tmp_path: Path) -> None:
    paths = resolve_paths(
        [
            fixtures_dir / "clean_minimal.py",
            fixtures_dir / "jl002_short_jit.py",
        ]
    )
    cfg = LintConfig()
    cache = tmp_path / "cache"
    cold = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True, jobs=2, cache_dir=cache)
    warm = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True, jobs=2, cache_dir=cache)
    assert cold == warm


@pytest.mark.integration
def test_cache_invalidates_when_file_changes(tmp_path: Path) -> None:
    p = tmp_path / "mod.py"
    p.write_text("def f():\n    '''A'''\n    pass\n", encoding="utf-8")
    cfg = LintConfig(select=frozenset(["JD"]))
    cache = tmp_path / "cache"

    run_checks([p], cfg=cfg, perf=False, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    run_checks([p], cfg=cfg, perf=False, doc=True, mathjx=True, jobs=1, cache_dir=cache)

    parses = {"n": 0}
    orig_parse = ast.parse

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    p.write_text(
        "def f():\n    '''A'''\n    pass\n\n# touch\n",
        encoding="utf-8",
    )
    with patch.object(ast, "parse", counting_parse):
        run_checks([p], cfg=cfg, perf=False, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    assert parses["n"] >= 1


@pytest.mark.integration
def test_cache_namespace_change_misses_cache(fixtures_dir: Path, tmp_path: Path) -> None:
    p = fixtures_dir / "jl002_short_jit.py"
    paths = resolve_paths([p])
    cache = tmp_path / "cache"
    cfg_jl = LintConfig(select=frozenset(["JL"]))
    cfg_empty = LintConfig(select=frozenset(["JM"]))  # no JL rules for this file content

    run_checks(paths, cfg=cfg_jl, perf=True, doc=False, mathjx=False, jobs=1, cache_dir=cache)
    ns_jl = namespace_segment(cfg_jl, perf=True, doc=False, mathjx=False)
    assert (cache / ns_jl).is_dir()

    out = run_checks(paths, cfg=cfg_empty, perf=True, doc=False, mathjx=False, jobs=1, cache_dir=cache)
    assert not any(d.rule_id.startswith("JL") for d in out)


@pytest.mark.integration
def test_corrupt_cache_entry_is_ignored(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text(
        'def g(x):\n    """Doc.\n\n    Parameters\n    ----------\n    x : int\n"""\n    return x\n',
        encoding="utf-8",
    )
    cfg = LintConfig()
    cache = tmp_path / "cache"
    run_checks([p], cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1, cache_dir=cache)

    for json_path in cache.rglob("*.json"):
        json_path.write_text("not-json {{{", encoding="utf-8")

    ok = run_checks([p], cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    assert isinstance(ok, list)


@pytest.mark.unit
def test_cache_second_run_skips_parse_when_warm(tmp_path: Path) -> None:
    p = tmp_path / "sample.py"
    p.write_text(
        'def f(x):\n    """Doc.\n\n    Parameters\n    ----------\n    x : int\n"""\n    return x\n',
        encoding="utf-8",
    )
    cfg = LintConfig()
    cache = tmp_path / "cache"
    parses = {"n": 0}
    orig_parse = ast.parse

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    with patch.object(ast, "parse", counting_parse):
        run_checks([p], cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    cold_parses = parses["n"]
    assert cold_parses >= 1

    parses["n"] = 0
    with patch.object(ast, "parse", counting_parse):
        run_checks([p], cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1, cache_dir=cache)
    assert parses["n"] == 0


@pytest.mark.smoke
def test_cli_check_accepts_cache_dir(clean_fixture: Path, tmp_path: Path) -> None:
    cache = tmp_path / "c"
    r = runner.invoke(app, ["check", "--no-doc", "--cache-dir", str(cache), str(clean_fixture)])
    assert r.exit_code == 0
    assert cache.is_dir()


@pytest.mark.unit
def test_load_config_cache_dir_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        """
[tool.jaxlint]
cache-dir = ".jaxlint-cache"
""".strip(),
        encoding="utf-8",
    )
    nested = root / "pkg"
    nested.mkdir()
    cfg = load_config(nested)
    assert cfg.cache_dir == (root / ".jaxlint-cache").resolve()
