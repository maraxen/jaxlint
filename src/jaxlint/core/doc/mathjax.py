"""MathJax / LaTeX light validation: JM001–JM003."""

from __future__ import annotations

import re
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def _strip_code_fences(text: str) -> str:
    out_lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out_lines.append(line)
    return "\n".join(out_lines)


def mathjax_diagnostics(doc: str, path: Path, *, line_base: int = 1) -> list[Diagnostic]:
    text = _strip_code_fences(doc)
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_delimiter_balance(text, path, line_base))
    diagnostics.extend(_latex_structure_scan(text, path, line_base))
    diagnostics.extend(_large_math_heuristic(doc, path, line_base))
    return diagnostics


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _delimiter_balance(full_text: str, path: Path, line_base: int) -> list[Diagnostic]:
    """Balance `$$` toggles and optional single `$` pairs (outside `$$`)."""
    diagnostics: list[Diagnostic] = []
    i = 0
    n = len(full_text)
    in_double = False
    single_count = 0
    double_start = -1
    while i < n:
        if full_text.startswith("$$", i):
            if in_double:
                in_double = False
                i += 2
                continue
            in_double = True
            double_start = i
            i += 2
            continue
        if not in_double and full_text[i] == "$":
            single_count += 1
        i += 1

    if in_double and double_start >= 0:
        diagnostics.append(
            Diagnostic(
                "JM001",
                Severity.ERROR,
                "Unterminated `$$` math block.",
                path,
                line_base + _line_for_offset(full_text, double_start) - 1,
            )
        )
    if single_count % 2 == 1:
        diagnostics.append(
            Diagnostic(
                "JM001",
                Severity.ERROR,
                "Unpaired inline `$` delimiter (odd count outside `$$`).",
                path,
                line_base,
            )
        )
    return diagnostics


def _latex_structure_scan(full_text: str, path: Path, line_base: int) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for m in re.finditer(r"\$\$(.*?)\$\$", full_text, re.DOTALL):
        body = m.group(1)
        line = line_base + _line_for_offset(full_text, m.start(1)) - 1
        err = _validate_latex_light(body)
        if err:
            diagnostics.append(Diagnostic("JM002", Severity.ERROR, err, path, line))
    return diagnostics


def _validate_latex_light(s: str) -> str | None:
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return "Unbalanced `}` in LaTeX block."
    if depth != 0:
        return "Unbalanced `{}` braces in LaTeX block."

    begins = list(re.finditer(r"\\begin\{([^}]*)\}", s))
    ends = list(re.finditer(r"\\end\{([^}]*)\}", s))
    if len(begins) != len(ends):
        return "`\\begin` / `\\end` environment count mismatch."
    for b, e in zip(begins, ends, strict=True):
        if b.group(1) != e.group(1):
            return f"Mismatched environment `{b.group(1)}` vs `{e.group(1)}`."

    if re.search(r"\\frac\b(?!\s*\{)", s):
        return r"`\\frac` should use `\\frac{numerator}{denominator}`."

    return None


def _large_math_heuristic(doc: str, path: Path, line_base: int) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    stripped = _strip_code_fences(doc)
    for m in re.finditer(r"\$\$(.*?)\$\$", stripped, re.DOTALL):
        body = (m.group(1) or "").strip()
        if len(body) <= 120:
            continue
        prose_outside = stripped.replace(m.group(0), " ", 1)
        letters = sum(1 for ch in prose_outside if ch.isalpha())
        if letters < 40:
            line = line_base + stripped[: m.start()].count("\n")
            out.append(
                Diagnostic(
                    "JM003",
                    Severity.WARNING,
                    "Large `$$...$$` block with little descriptive prose elsewhere.",
                    path,
                    line,
                )
            )
    return out
