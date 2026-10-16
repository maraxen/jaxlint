"""JL005 shape introspection inside JAX kernels (warning by default)."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def shape_diagnostics(tree: ast.Module, path: Path, *, strict: bool = False) -> list[Diagnostic]:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)  # noqa: B010
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
                    if _shape_feeds_shape_op(child):
                        continue
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


# Functions that accept a shape tuple and remain polymorphic under tracing.
_SHAPE_PROPAGATING_OPS = frozenset({
    "zeros", "ones", "full", "empty", "broadcast_to",
    "normal", "uniform", "bernoulli", "randint", "truncated_normal",
    "zeros_like", "ones_like",
})


def _shape_feeds_shape_op(node: ast.Attribute) -> bool:
    """Return True when this .shape attribute is a direct positional arg to a shape-propagating op.

    jax.random.normal(key, arr.shape, ...) and jnp.zeros(arr.shape) are the
    canonical idioms — arr.shape remains polymorphic because the consuming op
    accepts a shape tuple and does not branch on its value.
    """
    parent = getattr(node, "parent", None)
    if not isinstance(parent, ast.Call):
        return False
    func = parent.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _SHAPE_PROPAGATING_OPS:
        return False
    return node in parent.args
