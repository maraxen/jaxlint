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
            if not _is_dtype_kwarg(node) and not _is_host_consumed(node):
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
            if not _is_provably_nonzero(node.right):
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


def _is_provably_nonzero(node: ast.expr) -> bool:
    """Return True when divisor is statically bounded away from zero.

    Covers jnp.maximum(x, c) with c > 0, expr + eps with eps > 0, and
    Name nodes whose most-recent same-scope assignment satisfies either rule.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "maximum" and len(node.args) == 2:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value > 0:
                    return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        for operand in (node.left, node.right):
            if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)) and operand.value > 0:
                return True
    # If the divisor is a local variable, check its most recent assignment.
    if isinstance(node, ast.Name):
        defn = _find_most_recent_assign(node)
        if defn is not None:
            return _is_provably_nonzero(defn)
    return False


def _find_most_recent_assign(name_node: ast.Name) -> ast.expr | None:
    """Return the RHS of the most recent simple assignment to name_node in its enclosing body."""
    target = name_node.id
    curr: ast.AST | None = name_node
    while curr is not None:
        parent = getattr(curr, "parent", None)
        if parent is None:
            break
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            body = parent.body
            # Find the index of the statement containing curr
            stmt_ancestor = _stmt_ancestor(name_node, parent)
            if stmt_ancestor is None:
                break
            try:
                cut = body.index(stmt_ancestor)
            except ValueError:
                break
            # Scan backward for a simple assignment: target = <expr>
            for stmt in reversed(body[:cut]):
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == target
                ):
                    return stmt.value
            break
        curr = parent
    return None


def _stmt_ancestor(node: ast.AST, func_or_module: ast.AST) -> ast.stmt | None:
    """Return the direct-child statement of func_or_module that contains node."""
    body: list[ast.stmt] = getattr(func_or_module, "body", [])
    curr: ast.AST | None = node
    while curr is not None:
        if curr in body:
            return curr  # type: ignore[return-value]
        curr = getattr(curr, "parent", None)
    return None


def _is_dtype_kwarg(node: ast.AST) -> bool:
    """Return True when this node is the value of a dtype= keyword argument."""
    parent = getattr(node, "parent", None)
    return isinstance(parent, ast.keyword) and parent.arg == "dtype"


def _is_host_consumed(node: ast.AST, depth: int = 6) -> bool:
    """Return True when the node is inside a float()/int()/.item() host-side call."""
    curr = node
    for _ in range(depth):
        curr = getattr(curr, "parent", None)
        if curr is None:
            return False
        if isinstance(curr, ast.Call):
            func = curr.func
            if isinstance(func, ast.Name) and func.id in ("float", "int"):
                return True
            if isinstance(func, ast.Attribute) and func.attr == "item":
                return True
    return False
