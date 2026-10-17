"""Inline suppression comments: ``# jaxlint: ignore[JL003,JL012]`` or bare ``# jaxlint: ignore``.

Comments are found with :mod:`tokenize`, so a ``#`` inside a string literal is never a
comment. A suppression applies only to diagnostics reported on the same physical line.
Rule IDs are matched case-sensitively; unknown IDs are inert (no error, no diagnostic).
"""

from __future__ import annotations

import io
import re
import tokenize

from jaxlint.core.diagnostic import Diagnostic

_DIRECTIVE = re.compile(r"#\s*jaxlint:\s*ignore(?:\[(?P<ids>[^\]]*)\])?(?=\s|$)")

# line number -> set of rule IDs, or None meaning "all rules" (bare ignore).
Suppressions = dict[int, frozenset[str] | None]


def parse_suppressions(source: str) -> Suppressions:
    """Map 1-based line numbers to the rule IDs suppressed on that line."""
    out: Suppressions = {}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            m = _DIRECTIVE.match(tok.string.strip())
            if m is None:
                continue
            line = tok.start[0]
            ids_raw = m.group("ids")
            if ids_raw is None:
                out[line] = None
            else:
                ids = frozenset(s.strip() for s in ids_raw.split(",") if s.strip())
                out[line] = ids
    except (tokenize.TokenError, SyntaxError):
        # Unterminated constructs at EOF: keep the suppressions gathered so far.
        pass
    return out


def filter_suppressed(diagnostics: list[Diagnostic], source: str) -> list[Diagnostic]:
    """Drop diagnostics whose reported line carries a matching suppression comment."""
    if not diagnostics:
        return diagnostics
    suppressions = parse_suppressions(source)
    if not suppressions:
        return diagnostics
    kept: list[Diagnostic] = []
    for d in diagnostics:
        if d.line in suppressions:
            ids = suppressions[d.line]
            if ids is None or d.rule_id in ids:
                continue
        kept.append(d)
    return kept
