"""Docstring sections: JD001–JD005."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from jaxlint.core.config import LintConfig
from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.doc.mathjax import mathjax_diagnostics
from jaxlint.core.doc.style import StyleValidator

_GRIFFE_KIND_BY_TITLE: dict[str, str] = {
    "parameters": "parameters",
    "returns": "returns",
    "raises": "raises",
    "examples": "examples",
    "pseudocode": "pseudocode",
    "hardware notes": "hardware",
}


def _cfg_order_to_kinds(section_order: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for title in section_order:
        tl = title.strip().lower()
        for key, gk in _GRIFFE_KIND_BY_TITLE.items():
            if tl.startswith(key) or tl == key.rstrip("s"):
                out.append(gk)
                break
        else:
            if "pseudo" in tl:
                out.append("pseudocode")
            elif "hardware" in tl:
                out.append("hardware")
    return list(dict.fromkeys(out))


def _docstring_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].lineno
    return node.lineno


def doc_diagnostics_for_file(
    path: Path,
    cfg: LintConfig,
    *,
    sections: bool = True,
    mathjax: bool = True,
) -> list[Diagnostic]:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return []

    diagnostics: list[Diagnostic] = []
    validator = StyleValidator()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not (node.name.startswith("__") and node.name.endswith("__")):
                continue
            diagnostics.extend(
                _function_doc_diagnostics(node, path, cfg, validator, sections=sections, mathjax=mathjax)
            )
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            ds = ast.get_docstring(node, clean=False)
            if not ds or not ds.strip():
                diagnostics.append(
                    Diagnostic(
                        "JD001",
                        Severity.ERROR,
                        f"Class `{node.name}` has no docstring.",
                        path,
                        node.lineno,
                    )
                )
            elif mathjax:
                diagnostics.extend(mathjax_diagnostics(ds, path, line_base=_docstring_start_line(node)))
    return diagnostics


def _function_doc_diagnostics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    path: Path,
    cfg: LintConfig,
    validator: StyleValidator,
    *,
    sections: bool,
    mathjax: bool,
) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    doc_line = _docstring_start_line(node)
    doc = ast.get_docstring(node, clean=False)

    if not doc or not doc.strip():
        out.append(
            Diagnostic(
                "JD001",
                Severity.ERROR,
                f"Function `{node.name}` has no docstring.",
                path,
                node.lineno,
            )
        )
        return out

    if mathjax:
        out.extend(mathjax_diagnostics(doc, path, line_base=doc_line))

    if not sections:
        return out

    val = validator.validate(doc, cfg.docstring_style)
    kinds_present = val.get("sections") or []
    parsed = val.get("parsed_sections") or []

    for req_title in cfg.required_sections:
        rt = req_title.strip().lower()
        if "pseudo" in rt:
            if not re.search(r"^\s*Pseudocode\s*[:-]?", doc, re.MULTILINE | re.IGNORECASE):
                out.append(
                    Diagnostic(
                        "JD002",
                        Severity.ERROR,
                        f"Missing required `Pseudocode` section in `{node.name}`.",
                        path,
                        doc_line,
                    )
                )
            continue
        if "hardware" in rt:
            if "hardware notes" not in doc.lower():
                out.append(
                    Diagnostic(
                        "JD002",
                        Severity.ERROR,
                        f"Missing Hardware Notes in `{node.name}`.",
                        path,
                        doc_line,
                    )
                )
            continue
        if rt.startswith("param") or rt == "parameters":
            if not any(k in kinds_present for k in ("parameters", "args", "arguments")):
                out.append(
                    Diagnostic(
                        "JD002",
                        Severity.ERROR,
                        f"Missing Parameters/Args in `{node.name}`.",
                        path,
                        doc_line,
                    )
                )
            continue
        if rt.startswith("return"):
            if not any(k in kinds_present for k in ("returns", "return")):
                out.append(
                    Diagnostic(
                        "JD002",
                        Severity.ERROR,
                        f"Missing Returns in `{node.name}`.",
                        path,
                        doc_line,
                    )
                )

    kind_rank = {k: i for i, k in enumerate(_cfg_order_to_kinds(cfg.section_order))}
    ordered_seen: list[tuple[str, int]] = []
    approximate_line = doc_line
    for sec in parsed:
        k = sec.kind.value if hasattr(sec.kind, "value") else str(sec.kind)
        if k in kind_rank:
            ordered_seen.append((k, approximate_line))
        approximate_line += len(str(getattr(sec, "value", "")).splitlines()) + 1

    rank_seq = [kind_rank[k] for k, _ in ordered_seen]
    if rank_seq != sorted(rank_seq) and len(rank_seq) >= 2:
        out.append(
            Diagnostic(
                "JD003",
                Severity.WARNING,
                f"Sections not in configured canonical order in `{node.name}`.",
                path,
                doc_line,
            )
        )

    doc_names = _documented_parameter_names(parsed)
    code_names = _code_parameter_names(node)
    if doc_names:
        skip = {"self", "cls"}
        undocumented = sorted(set(code_names) - set(doc_names) - skip)
        extras = sorted(set(doc_names) - set(code_names))
        if undocumented:
            out.append(
                Diagnostic(
                    "JD004",
                    Severity.WARNING,
                    f"Parameters missing from doc: {', '.join(undocumented)} in `{node.name}`.",
                    path,
                    doc_line,
                )
            )
        if extras:
            out.append(
                Diagnostic(
                    "JD004",
                    Severity.WARNING,
                    f"Doc mentions unknown parameters: {', '.join(extras)} in `{node.name}`.",
                    path,
                    doc_line,
                )
            )

    if node.returns is not None and not _is_none_expr(node.returns):
        has_ret = any(k in kinds_present for k in ("returns", "return"))
        if not has_ret:
            out.append(
                Diagnostic(
                    "JD005",
                    Severity.WARNING,
                    f"Return annotation present but no Returns section in `{node.name}`.",
                    path,
                    doc_line,
                )
            )

    return out


def _documented_parameter_names(parsed_sections: list) -> set[str]:
    names: set[str] = set()
    for sec in parsed_sections:
        k = sec.kind.value if hasattr(sec.kind, "value") else str(sec.kind)
        if k in ("parameters", "args", "arguments"):
            for p in sec.value:
                pname = getattr(p, "name", None)
                if pname:
                    names.add(str(pname))
    return names


def _code_parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    names: list[str] = []
    names.extend(a.arg for a in args.posonlyargs)
    names.extend(a.arg for a in args.args)
    if args.vararg:
        names.append(args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return set(names)


def _is_none_expr(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None
