"""JL005 shape introspection inside JAX kernels (warning by default)."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def shape_diagnostics(tree: ast.Module, path: Path, *, strict: bool = False) -> list[Diagnostic]:
    v = ShapeSensor(path, strict)
    v.visit(tree)
    return v.diagnostics


class ShapeSensor(ast.NodeVisitor):
    def __init__(self, path: Path, strict: bool):
        self.path = path
        self.strict = strict
        self.diagnostics: list[Diagnostic] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_jax_kernel(node.decorator_list):
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr == "shape":
                    sev = Severity.ERROR if self.strict else Severity.WARNING
                    self.diagnostics.append(
                        Diagnostic(
                            "JL005",
                            sev,
                            "`.shape` in JAX kernel — may break polymorphic shapes.",
                            self.path,
                            child.lineno,
                        )
                    )
                elif isinstance(child, ast.Call):
                    fn = child.func
                    if isinstance(fn, ast.Attribute) and fn.attr == "shape":
                        sev = Severity.ERROR if self.strict else Severity.WARNING
                        self.diagnostics.append(
                            Diagnostic(
                                "JL005",
                                sev,
                                "`jnp.shape()` / `.shape`-like call in JAX kernel.",
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
