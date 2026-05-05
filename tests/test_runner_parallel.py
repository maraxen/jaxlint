"""Parallel ``run_checks`` must match serial results (sorted diagnostics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.runner import resolve_paths, run_checks


@pytest.mark.integration
def test_parallel_jobs_match_serial_on_fixtures(fixtures_dir: Path) -> None:
    paths = resolve_paths(
        [
            fixtures_dir / "clean_minimal.py",
            fixtures_dir / "jl001_scatter_gather.py",
            fixtures_dir / "jd001_missing_docstring.py",
        ]
    )
    cfg = LintConfig()
    serial = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1)
    for j in (2, 4):
        parallel = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True, jobs=j)
        assert parallel == serial
        assert parallel == sorted(parallel, key=lambda d: d.sort_key())


@pytest.mark.integration
def test_parallel_perf_only_matches_serial(fixtures_dir: Path) -> None:
    paths = resolve_paths([fixtures_dir / "jl002_short_jit.py", fixtures_dir / "jl010_float64.py"])
    cfg = LintConfig(select=frozenset(["JL"]))
    serial = run_checks(paths, cfg=cfg, perf=True, doc=False, mathjx=False, jobs=1)
    parallel = run_checks(paths, cfg=cfg, perf=True, doc=False, mathjx=False, jobs=2)
    assert parallel == serial


@pytest.mark.integration
def test_parallel_doc_only_matches_serial(fixtures_dir: Path) -> None:
    paths = resolve_paths(
        [
            fixtures_dir / "jd001_missing_docstring.py",
            fixtures_dir / "clean_minimal.py",
        ]
    )
    cfg = LintConfig(select=frozenset(["JD", "JM"]))
    serial = run_checks(paths, cfg=cfg, perf=False, doc=True, mathjx=True, jobs=1)
    parallel = run_checks(paths, cfg=cfg, perf=False, doc=True, mathjx=True, jobs=2)
    assert parallel == serial
