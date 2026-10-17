"""JL023 per-iteration host sync on a device value (info; ``src/`` paths by default).

A host-side ``for``/``while`` body that forces a device value to the host on every
iteration (``.item()``, ``float(x)``, ``np.asarray(x)``, ``block_until_ready``, ...) serialises
the device against Python once per trip. The sensor is AST-only and flow-insensitive within
a scope: a name is device-tainted when any assignment in the same scope binds it to a device
expression. Loops whose body reads a wall clock (``time.perf_counter``, ``timeit``) are benchmark
loops, where a sync is intentional, and are skipped.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.sensors._jax_names import (
    KERNEL_FAMILY,
    alias_map,
    is_kernel_decorator,
    jitted_name_table,
    resolve_dotted,
)

RULE_ID = "JL023"

_SYNC_METHODS = frozenset({"item", "tolist", "block_until_ready"})
_SCALAR_CASTS = frozenset({"float", "int", "bool"})
_NUMPY_SYNCS = {"numpy.asarray": "np.asarray()", "numpy.array": "np.array()"}
_JAX_SYNCS = {
    "jax.device_get": "jax.device_get()",
    "jax.block_until_ready": "jax.block_until_ready()",
}
# jax.* calls that return host values or callables, not device arrays.
_JAX_NON_DEVICE = KERNEL_FAMILY | frozenset(
    {
        "jax.device_get",
        "jax.devices",
        "jax.local_devices",
        "jax.device_count",
        "jax.local_device_count",
        "jax.process_index",
        "jax.process_count",
    }
)
# Attributes of a device array that are host metadata, not device values.
_METADATA_ATTRS = frozenset({"shape", "ndim", "dtype", "size"})
_TIMING_CALLS = frozenset({"time.perf_counter", "time.perf_counter_ns", "time.time"})
_SCOPE_BARRIERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def host_sync_diagnostics(
    tree: ast.Module, path: Path, *, everywhere: bool = False
) -> list[Diagnostic]:
    """JL023 diagnostics for ``tree``; outside ``src/`` paths only when ``everywhere`` is set."""
    if not everywhere and "src" not in path.parts:
        return []
    aliases = alias_map(tree)
    jit_table = jitted_name_table(tree, aliases)
    out: list[Diagnostic] = []
    for body in _host_scope_bodies(tree, aliases):
        out.extend(_scope_diagnostics(body, path, aliases, jit_table))
    return out


def _host_scope_bodies(tree: ast.Module, aliases: dict[str, str]) -> list[list[ast.stmt]]:
    """Statement lists of the module and of every function that is not a kernel or inside one."""
    bodies: list[list[ast.stmt]] = [tree.body]

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(is_kernel_decorator(d, aliases) for d in child.decorator_list):
                    continue
                bodies.append(child.body)
            visit(child)

    visit(tree)
    return bodies


def _scope_walk(stmts: list[ast.stmt]) -> Iterator[ast.AST]:
    """Yield every node under ``stmts`` without descending into nested scopes."""
    # The barrier applies to the starting statements too: a module body must not walk
    # into its top-level functions, which are scanned as scopes of their own.
    stack: list[ast.AST] = [s for s in stmts if not isinstance(s, _SCOPE_BARRIERS)]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _SCOPE_BARRIERS):
                stack.append(child)


def _scope_diagnostics(
    body: list[ast.stmt], path: Path, aliases: dict[str, str], jit_table: dict[str, str]
) -> list[Diagnostic]:
    tainted = _device_names(body, aliases, jit_table)
    out: list[Diagnostic] = []
    seen: set[int] = set()
    for loop in (n for n in _scope_walk(body) if isinstance(n, _LOOPS)):
        loop_nodes = list(_scope_walk(loop.body))
        if _times_the_loop(loop_nodes, aliases):
            continue
        for call in (n for n in loop_nodes if isinstance(n, ast.Call)):
            if id(call) in seen:
                continue
            hit = _sync_call(call, tainted, aliases, jit_table)
            if hit is None:
                continue
            seen.add(id(call))
            op, name = hit
            msg = (
                f"`{op}` on device value `{name}` inside a loop forces a host sync every "
                "iteration; keep values on device across the loop and transfer once."
            )
            out.append(Diagnostic(RULE_ID, Severity.INFO, msg, path, call.lineno, call.col_offset))
    return out


def _times_the_loop(nodes: list[ast.AST], aliases: dict[str, str]) -> bool:
    for node in nodes:
        if isinstance(node, ast.Call):
            head = resolve_dotted(node.func, aliases)
            if head and (head in _TIMING_CALLS or head.startswith("timeit.")):
                return True
    return False


def _device_names(
    body: list[ast.stmt], aliases: dict[str, str], jit_table: dict[str, str]
) -> set[str]:
    """Names bound in this scope to a device expression (fixpoint over assignments)."""
    assigns = [
        n for n in _scope_walk(body) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
    ]
    assigns.sort(key=lambda n: n.lineno)
    tainted: set[str] = set()
    changed = True
    while changed:
        before = len(tainted)
        for node in assigns:
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                _taint_target(target, value, tainted, aliases, jit_table)
        changed = len(tainted) != before
    return tainted


def _taint_target(
    target: ast.expr,
    value: ast.expr,
    tainted: set[str],
    aliases: dict[str, str],
    jit_table: dict[str, str],
) -> None:
    if isinstance(target, ast.Name):
        if _is_device(value, tainted, aliases, jit_table):
            tainted.add(target.id)
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
            for t, v in zip(target.elts, value.elts, strict=True):
                _taint_target(t, v, tainted, aliases, jit_table)
        elif _is_device(value, tainted, aliases, jit_table):
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    tainted.add(node.id)


def _is_device(
    expr: ast.expr, tainted: set[str], aliases: dict[str, str], jit_table: dict[str, str]
) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in tainted
    if isinstance(expr, ast.Subscript):
        return _is_device(expr.value, tainted, aliases, jit_table)
    if isinstance(expr, ast.Attribute):
        return expr.attr not in _METADATA_ATTRS and _is_device(
            expr.value, tainted, aliases, jit_table
        )
    if isinstance(expr, ast.Call):
        head = resolve_dotted(expr.func, aliases)
        if head is None:
            return False
        if head in jit_table:
            return True
        if head.startswith("jax.") and head not in _JAX_NON_DEVICE:
            return True
    return False


def _sync_call(
    call: ast.Call, tainted: set[str], aliases: dict[str, str], jit_table: dict[str, str]
) -> tuple[str, str] | None:
    """Return ``(op, value-name)`` when ``call`` forces a device value to the host."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in _SYNC_METHODS:
        if _is_device(func.value, tainted, aliases, jit_table):
            return f".{func.attr}()", _describe(func.value)
    if isinstance(func, ast.Name) and func.id in _SCALAR_CASTS and len(call.args) == 1:
        arg = call.args[0]
        if _is_device(arg, tainted, aliases, jit_table):
            return f"{func.id}()", _describe(arg)
        return None
    head = resolve_dotted(func, aliases)
    op = _NUMPY_SYNCS.get(head or "") or _JAX_SYNCS.get(head or "")
    if op is not None and call.args and _is_device(call.args[0], tainted, aliases, jit_table):
        return op, _describe(call.args[0])
    return None


def _describe(expr: ast.expr) -> str:
    node = expr
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    text = ast.unparse(expr)
    return text if len(text) <= 40 else text[:37] + "..."
