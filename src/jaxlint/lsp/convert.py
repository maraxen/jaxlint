"""Map :class:`jaxlint.core.diagnostic.Diagnostic` to LSP structures."""

from __future__ import annotations

from lsprotocol import types

from jaxlint.core.diagnostic import Diagnostic, Severity


def jaxlint_to_lsp_diagnostic(d: Diagnostic) -> types.Diagnostic:
    """Convert a jaxlint diagnostic to LSP ``Diagnostic``."""

    start_line = max(0, d.line - 1)
    start_char = 0 if d.col <= 0 else d.col - 1
    end_line_raw = d.end_line if d.end_line is not None else d.line
    end_line = max(0, end_line_raw - 1)

    if d.end_col is not None:
        end_char = 0 if d.end_col <= 0 else d.end_col
    elif end_line == start_line:
        end_char = start_char + 1
    else:
        end_char = 0

    sev = types.DiagnosticSeverity.Hint
    if d.severity == Severity.ERROR:
        sev = types.DiagnosticSeverity.Error
    elif d.severity == Severity.WARNING:
        sev = types.DiagnosticSeverity.Warning
    elif d.severity == Severity.INFO:
        sev = types.DiagnosticSeverity.Information

    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=start_line, character=start_char),
            end=types.Position(line=end_line, character=end_char),
        ),
        message=d.message,
        severity=sev,
        code=d.rule_id,
        source="jaxlint",
    )


def jaxlint_diagnostics_to_lsp(diagnostics: list[Diagnostic]) -> list[types.Diagnostic]:
    """Convert a list of jaxlint diagnostics to LSP diagnostics."""

    return [jaxlint_to_lsp_diagnostic(d) for d in diagnostics]
