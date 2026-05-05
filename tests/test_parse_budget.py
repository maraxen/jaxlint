"""Assert read/parse budget through ``run_checks`` and thin wrappers."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.doc.sections import doc_diagnostics_for_file
from jaxlint.core.runner import run_checks
from jaxlint.core.sensors import perf_diagnostics_for_file


@pytest.fixture
def minimal_py(tmp_path: Path) -> Path:
    p = tmp_path / "sample.py"
    p.write_text(
        'def f(x):\n    """Doc.\n\n    Parameters\n    ----------\n    x : int\n"""\n    return x\n',
        encoding="utf-8",
    )
    return p


@pytest.mark.unit
def test_run_checks_perf_only_single_read_single_parse(minimal_py: Path) -> None:
    reads = {"n": 0}
    parses = {"n": 0}
    orig_read = Path.read_text
    orig_parse = ast.parse

    def counting_read(self: Path, *a, **k):
        reads["n"] += 1
        return orig_read(self, *a, **k)

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    cfg = LintConfig(select=frozenset(["JL"]))
    with patch.object(Path, "read_text", counting_read):
        with patch.object(ast, "parse", counting_parse):
            run_checks([minimal_py], cfg=cfg, perf=True, doc=False, mathjx=False, jobs=1)
    assert reads["n"] == 1
    assert parses["n"] == 1


@pytest.mark.unit
def test_run_checks_doc_only_single_read_single_parse(minimal_py: Path) -> None:
    reads = {"n": 0}
    parses = {"n": 0}
    orig_read = Path.read_text
    orig_parse = ast.parse

    def counting_read(self: Path, *a, **k):
        reads["n"] += 1
        return orig_read(self, *a, **k)

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    cfg = LintConfig(select=frozenset(["JD"]))
    with patch.object(Path, "read_text", counting_read):
        with patch.object(ast, "parse", counting_parse):
            run_checks([minimal_py], cfg=cfg, perf=False, doc=True, mathjx=True, jobs=1)
    assert reads["n"] == 1
    assert parses["n"] == 1


@pytest.mark.unit
def test_run_checks_perf_and_doc_single_read_single_parse(minimal_py: Path) -> None:
    reads = {"n": 0}
    parses = {"n": 0}
    orig_read = Path.read_text
    orig_parse = ast.parse

    def counting_read(self: Path, *a, **k):
        reads["n"] += 1
        return orig_read(self, *a, **k)

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    cfg = LintConfig()
    with patch.object(Path, "read_text", counting_read):
        with patch.object(ast, "parse", counting_parse):
            run_checks([minimal_py], cfg=cfg, perf=True, doc=True, mathjx=True, jobs=1)
    assert reads["n"] == 1
    assert parses["n"] == 1


@pytest.mark.unit
def test_perf_diagnostics_for_file_matches_budget(minimal_py: Path) -> None:
    reads = {"n": 0}
    parses = {"n": 0}
    orig_read = Path.read_text
    orig_parse = ast.parse

    def counting_read(self: Path, *a, **k):
        reads["n"] += 1
        return orig_read(self, *a, **k)

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    cfg = LintConfig()
    with patch.object(Path, "read_text", counting_read):
        with patch.object(ast, "parse", counting_parse):
            perf_diagnostics_for_file(minimal_py, cfg)
    assert reads["n"] == 1
    assert parses["n"] == 1


@pytest.mark.unit
def test_doc_diagnostics_for_file_matches_budget(minimal_py: Path) -> None:
    reads = {"n": 0}
    parses = {"n": 0}
    orig_read = Path.read_text
    orig_parse = ast.parse

    def counting_read(self: Path, *a, **k):
        reads["n"] += 1
        return orig_read(self, *a, **k)

    def counting_parse(*a, **k):
        parses["n"] += 1
        return orig_parse(*a, **k)

    cfg = LintConfig()
    with patch.object(Path, "read_text", counting_read):
        with patch.object(ast, "parse", counting_parse):
            doc_diagnostics_for_file(minimal_py, cfg, sections=True, mathjax=True)
    assert reads["n"] == 1
    assert parses["n"] == 1
