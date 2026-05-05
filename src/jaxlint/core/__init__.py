"""Public core API surface."""

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.rules import RULES, describe_rules
from jaxlint.core.runner import format_json, format_text, resolve_paths, run_checks

__all__ = [
    "Diagnostic",
    "Severity",
    "RULES",
    "describe_rules",
    "resolve_paths",
    "run_checks",
    "format_text",
    "format_json",
]
