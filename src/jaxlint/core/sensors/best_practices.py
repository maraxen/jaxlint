"""JL010 float64, JL011 ShardingSpec, JL012 division."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def best_practices_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)  # noqa: B010
    v = BestPracticesSensor(path)
    v.visit(tree)
    return v.diagnostics


class BestPracticesSensor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.diagnostics: list[Diagnostic] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in ("jnp", "jax_np") and node.attr == "float64":
            self.diagnostics.append(
                Diagnostic(
                    "JL010",
                    Severity.WARNING,
                    "`jnp.float64` — prefer f32/bf16 unless double precision required.",
                    self.path,
                    node.lineno,
                )
            )
        if node.attr == "ShardingSpec":
            self.diagnostics.append(
                Diagnostic(
                    "JL011",
                    Severity.WARNING,
                    "`ShardingSpec` — prefer NamedSharding + Mesh.",
                    self.path,
                    node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div) and _is_inside_kernel(node):
            self.diagnostics.append(
                Diagnostic(
                    "JL012",
                    Severity.WARNING,
                    "Division `/` in JIT kernel — use safe reciprocal if divisor may be zero.",
                    self.path,
                    node.lineno,
                )
            )
        self.generic_visit(node)


def _is_inside_kernel(node: ast.AST) -> bool:
    curr: ast.AST | None = node
    while curr:
        if isinstance(curr, ast.FunctionDef):
            return any(_has_jit_in_str(d) for d in curr.decorator_list)
        curr = getattr(curr, "parent", None)
    return False


def _has_jit_in_str(d: ast.expr) -> bool:
    return "jit" in ast.unparse(d)
