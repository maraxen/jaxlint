"""JL003 loops in JAX kernels."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.sensors._jax_names import alias_map, is_kernel_decorator


def loop_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    visitor = LoopSensor(path)
    visitor.visit(tree)
    return visitor.diagnostics


class LoopSensor(ast.NodeVisitor):
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
