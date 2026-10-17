"""JL004 side effects in JAX kernels."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.sensors._jax_names import alias_map, is_kernel_decorator


def purity_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    v = PuritySensor(path)
    v.visit(tree)
    return v.diagnostics


class PuritySensor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.diagnostics: list[Diagnostic] = []
        self.aliases: dict[str, str] = {}

    def visit_Module(self, node: ast.Module) -> None:
        self.aliases = alias_map(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if any(is_kernel_decorator(d, self.aliases) for d in node.decorator_list):
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
