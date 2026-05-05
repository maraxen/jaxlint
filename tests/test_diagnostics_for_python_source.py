"""Tests for :func:`jaxlint.core.runner.diagnostics_for_python_source`."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.config import load_config
from jaxlint.core.runner import diagnostics_for_python_source, run_checks


@pytest.fixture
def jd001_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "jd001_missing_docstring.py"


def test_in_memory_matches_run_checks(jd001_path: Path, tmp_path: Path) -> None:
    text = jd001_path.read_text(encoding="utf-8")
    dst = tmp_path / "jd001_missing_docstring.py"
    dst.write_text(text, encoding="utf-8")
    cfg = load_config(tmp_path)
    disk = run_checks([dst], cfg=cfg)
    mem = diagnostics_for_python_source(dst, text, cfg=cfg)
    assert [d.sort_key() for d in disk] == [d.sort_key() for d in mem]
    assert [d.rule_id for d in disk] == [d.rule_id for d in mem]


def test_syntax_error_returns_empty_when_perf_and_doc_enabled(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    cfg = load_config(tmp_path)
    out = diagnostics_for_python_source(
        path,
        "def broken(\n",
        cfg=cfg,
        perf=True,
        doc=True,
        mathjx=True,
    )
    assert out == []


def test_clean_file_no_diagnostics(clean_fixture: Path, tmp_path: Path) -> None:
    text = clean_fixture.read_text(encoding="utf-8")
    dst = tmp_path / "clean.py"
    dst.write_text(text, encoding="utf-8")
    cfg = load_config(tmp_path)
    mem = diagnostics_for_python_source(dst, text, cfg=cfg)
    disk = run_checks([dst], cfg=cfg)
    assert mem == disk
