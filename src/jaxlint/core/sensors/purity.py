"""JL004 side effects in JAX kernels."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def purity_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    v = PuritySensor(path)
    v.visit(tree)
    return v.diagnostics


class PuritySensor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.diagnostics: list[Diagnostic] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_jax_kernel(node.decorator_list):
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id in ("print", "open"):
                        self.diagnostics.append(
                            Diagnostic(
                                "JL004",
                                Severity.ERROR,
                                f"Side effect `{child.func.id}()` in JAX kernel.",
                                self.path,
                                child.lineno,
                            )
                        )
        self.generic_visit(node)


def _is_jax_kernel(dec_list: list[ast.expr]) -> bool:
    return any(_dec_jit_grad(d) for d in dec_list)


def _dec_jit_grad(d: ast.expr) -> bool:
    if isinstance(d, ast.Attribute) and d.attr in ("jit", "grad"):
        return True
    return isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in (
        "jit",
        "grad",
    )
