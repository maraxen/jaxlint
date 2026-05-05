"""JL003 loops in JAX kernels."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def loop_diagnostics(code: str, path: Path) -> list[Diagnostic]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    visitor = LoopSensor(path)
    visitor.visit(tree)
    return visitor.diagnostics


class LoopSensor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.diagnostics: list[Diagnostic] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_jax_kernel(node.decorator_list):
            for child in ast.walk(node):
                if isinstance(child, (ast.For, ast.While)):
                    self.diagnostics.append(
                        Diagnostic(
                            "JL003",
                            Severity.ERROR,
                            "Loop in JAX kernel — use jax.lax.scan / fori_loop.",
                            self.path,
                            child.lineno,
                        )
                    )
        self.generic_visit(node)


def _is_jax_kernel(dec_list: list[ast.expr]) -> bool:
    return any(_dec_is_grad_or_jit(d) for d in dec_list)


def _dec_is_grad_or_jit(dec: ast.AST) -> bool:
    if isinstance(dec, ast.Attribute) and dec.attr in ("jit", "grad"):
        return True
    return isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in (
        "jit",
        "grad",
    )
