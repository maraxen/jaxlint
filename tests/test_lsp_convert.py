"""LSP diagnostic conversion (requires ``jaxlint[lsp]``)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lsprotocol")

from lsprotocol import types

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.lsp.convert import jaxlint_to_lsp_diagnostic


def test_severity_mapping() -> None:
    d = Diagnostic("R", Severity.ERROR, "e", Path("a.py"), 1, 1)
    assert jaxlint_to_lsp_diagnostic(d).severity == types.DiagnosticSeverity.Error
    d = Diagnostic("R", Severity.WARNING, "w", Path("a.py"), 1, 1)
    assert jaxlint_to_lsp_diagnostic(d).severity == types.DiagnosticSeverity.Warning
    d = Diagnostic("R", Severity.INFO, "i", Path("a.py"), 1, 1)
    assert jaxlint_to_lsp_diagnostic(d).severity == types.DiagnosticSeverity.Information


def test_range_single_line() -> None:
    d = Diagnostic("JL001", Severity.ERROR, "m", Path("x.py"), line=2, col=3)
    ld = jaxlint_to_lsp_diagnostic(d)
    assert ld.range.start.line == 1
    assert ld.range.start.character == 2
    assert ld.range.end.line == 1
    assert ld.range.end.character == 3


def test_col_zero_maps_to_line_start() -> None:
    d = Diagnostic("JL001", Severity.ERROR, "m", Path("x.py"), line=1, col=0)
    ld = jaxlint_to_lsp_diagnostic(d)
    assert ld.range.start.character == 0


def test_code_and_source() -> None:
    d = Diagnostic("JD001", Severity.ERROR, "m", Path("x.py"), 1, 1)
    ld = jaxlint_to_lsp_diagnostic(d)
    assert ld.code == "JD001"
    assert ld.source == "jaxlint"


def test_multiline_span() -> None:
    d = Diagnostic(
        "JL099",
        Severity.WARNING,
        "m",
        Path("x.py"),
        line=1,
        col=1,
        end_line=3,
        end_col=5,
    )
    ld = jaxlint_to_lsp_diagnostic(d)
    assert ld.range.start.line == 0
    assert ld.range.end.line == 2
    assert ld.range.end.character == 5
