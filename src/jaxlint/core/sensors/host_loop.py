"""JL020: a host ``for`` loop dispatches a jitted function or ``jnp`` op per element and collects.

AST-only. A candidate call is either a module-local JIT-family name (``jitted_name_table``,
local file only) or a ``jnp.*`` function (``alias_map``). The loop is flagged when a candidate
call's result reaches a collection (``lst.append(...)``, ``lst.extend(...)``, ``arr[i] = ...``),
directly or through names assigned from it inside the loop body.

Guards (no flag):
- loops inside a kernel (JIT/GRAD-family function): that is not host code;
- carry recurrences, where an argument of the call is itself derived from the call's result
  (``x = step(x)``): serial by construction, the fix is ``lax.scan`` (a different rule);
- timing loops, whose body calls ``time.perf_counter``/``time.time``/``timeit`` or
  ``block_until_ready``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.sensors._jax_names import (
    alias_map,
    is_kernel_decorator,
    jitted_name_table,
    resolve_dotted,
)

_TIMING_CALLS: frozenset[str] = frozenset(
    {"time.perf_counter", "time.perf_counter_ns", "time.time", "time.time_ns", "time.monotonic"}
)
_COLLECT_METHODS: frozenset[str] = frozenset({"append", "extend"})
_NOT_DIRECT: tuple[type[ast.AST], ...] = (
    ast.For,
    ast.While,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def host_loop_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    visitor = _HostLoopSensor(path, alias_map(tree), jitted_name_table(tree, alias_map(tree)))
    visitor.visit(tree)
    return visitor.diagnostics


class _HostLoopSensor(ast.NodeVisitor):
    def __init__(self, path: Path, aliases: dict[str, str], jitted: dict[str, str]):
        self.path = path
        self.aliases = aliases
        self.jitted = jitted
        self.kernel_depth = 0
        self.diagnostics: list[Diagnostic] = []

    def _enter_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kernel = any(is_kernel_decorator(d, self.aliases) for d in node.decorator_list)
        self.kernel_depth += int(kernel)
        self.generic_visit(node)
        self.kernel_depth -= int(kernel)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_function(node)

    def visit_For(self, node: ast.For) -> None:
        if self.kernel_depth == 0:
            callee = self._flagged_callee(node)
            if callee is not None:
                self.diagnostics.append(
                    Diagnostic(
                        "JL020",
                        Severity.INFO,
                        f"Loop calls `{callee}` once per element and collects results; "
                        "dispatch is per element — batch with jax.vmap over the loop axis.",
                        self.path,
                        node.lineno,
                    )
                )
        self.generic_visit(node)

    def _callee_name(self, call: ast.Call) -> str | None:
        func = call.func
        if isinstance(func, ast.Name) and func.id in self.jitted:
            return func.id
        dotted = resolve_dotted(func, self.aliases)
        if dotted is not None and (dotted == "jax.numpy" or dotted.startswith("jax.numpy.")):
            return ast.unparse(func)
        return None

    def _flagged_callee(self, loop: ast.For) -> str | None:
        if _has_timing(loop):
            return None
        nodes = list(_direct_nodes(loop.body))
        assigns = [
            (_target_names(t), n.value)
            for n in nodes
            if isinstance(n, ast.Assign)
            for t in n.targets
        ]
        assigns += [
            (_target_names(n.target), n.value)
            for n in nodes
            if isinstance(n, ast.AnnAssign) and n.value is not None
        ]
        candidates = [
            n for n in nodes if isinstance(n, ast.Call) and self._callee_name(n) is not None
        ]
        candidates.sort(key=lambda c: (c.lineno, c.col_offset))
        for call in candidates:
            tainted = _taint_from(call, assigns)
            if _carries(call, tainted):
                continue
            if _collects(nodes, call, tainted):
                return ast.unparse(call.func)
        return None


def _direct_nodes(stmts: list[ast.stmt]) -> Iterator[ast.AST]:
    """Yield the nodes of ``stmts`` without descending into nested loops, defs or lambdas."""
    stack: list[ast.AST] = [s for s in stmts if not isinstance(s, _NOT_DIRECT)]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _NOT_DIRECT):
                stack.append(child)


def _has_timing(loop: ast.For) -> bool:
    for node in ast.walk(loop):
        if isinstance(node, ast.Attribute) and node.attr == "block_until_ready":
            return True
        if isinstance(node, ast.Call):
            head = _dotted_of(node.func)
            if head in _TIMING_CALLS or (head is not None and head.startswith("timeit.")):
                return True
    return False


def _dotted_of(expr: ast.expr) -> str | None:
    parts: list[str] = []
    node = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in target.elts:
            out |= _target_names(elt)
        return out
    return set()


def _mentions(expr: ast.AST, call: ast.Call, tainted: set[str]) -> bool:
    for node in ast.walk(expr):
        if node is call:
            return True
        if isinstance(node, ast.Name) and node.id in tainted:
            return True
    return False


def _taint_from(call: ast.Call, assigns: list[tuple[set[str], ast.expr]]) -> set[str]:
    """Names whose value is (transitively) derived from ``call``'s result inside the loop."""
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for targets, value in assigns:
            if targets - tainted and _mentions(value, call, tainted):
                tainted |= targets
                changed = True
    return tainted


def _carries(call: ast.Call, tainted: set[str]) -> bool:
    """True when an argument of ``call`` is derived from ``call``'s own result (a recurrence)."""
    arg_nodes: list[ast.AST] = [*call.args, *(kw.value for kw in call.keywords)]
    for arg in arg_nodes:
        for node in ast.walk(arg):
            if isinstance(node, ast.Name) and node.id in tainted:
                return True
    return False


def _collects(nodes: list[ast.AST], call: ast.Call, tainted: set[str]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            method = node.value.func
            if (
                isinstance(method, ast.Attribute)
                and method.attr in _COLLECT_METHODS
                and isinstance(method.value, ast.Name)
                and any(_mentions(arg, call, tainted) for arg in node.value.args)
            ):
                return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
            if _mentions(node.value, call, tainted):
                return True
    return False
