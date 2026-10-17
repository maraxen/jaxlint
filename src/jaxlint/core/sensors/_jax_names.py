"""Shared, AST-only resolution of JAX kernel decorators (no JAX runtime, no cross-file lookup).

Used by the JL003 / JL004 / JL005 sensors. A "kernel" is a function decorated by a
transform in the JIT family or the GRAD family below. Local names are resolved through the
module's own imports, so ``@jit`` after ``from jax import jit``, ``@eqx.filter_jit`` after
``import equinox as eqx`` and ``@functools.partial(jax.jit, ...)`` are all recognised, while a
user function that happens to be called ``jit`` is not.

Public API
----------
alias_map(tree) -> dict[str, str]
    Local bound name -> canonical dotted module/object path, from every ``import`` and
    ``from ... import`` in the module (absolute imports only).
resolve_dotted(expr, aliases) -> str | None
    Canonical dotted name of a ``Name``/``Attribute`` chain; unbound leading names pass
    through unchanged, so ``jax.jit`` resolves to ``"jax.jit"`` even without an import.
is_kernel_decorator(node, aliases) -> bool
    True when a decorator expression applies a JIT- or GRAD-family transform, in bare,
    call, ``partial(...)`` or ``functools.partial(...)`` form.
jitted_name_table(tree, aliases) -> dict[str, str]
    Module-local names bound to a JIT-family wrapper, by decorator or by assignment
    (``f_jit = jax.jit(f)``, ``eqx.filter_jit(f)``, ``partial(jax.jit, ...)(f)``). Maps the
    bound name to the canonical wrapper (``"jax.jit"`` or ``"equinox.filter_jit"``).
    Scope-insensitive and local-file only: names are not resolved across files.
"""

from __future__ import annotations

import ast

JIT_FAMILY: frozenset[str] = frozenset(
    {"jax.jit", "jax.pmap", "equinox.filter_jit", "equinox.filter_pmap"}
)
GRAD_FAMILY: frozenset[str] = frozenset(
    {
        "jax.grad",
        "jax.value_and_grad",
        "equinox.filter_grad",
        "equinox.filter_value_and_grad",
    }
)
KERNEL_FAMILY: frozenset[str] = JIT_FAMILY | GRAD_FAMILY
_PARTIAL: str = "functools.partial"


def alias_map(tree: ast.Module) -> dict[str, str]:
    """Map each name bound by an absolute import in ``tree`` to its canonical dotted path."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
                else:
                    top = item.name.split(".", 1)[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            for item in node.names:
                if item.name == "*":
                    continue
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def resolve_dotted(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    """Return the canonical dotted name of a ``Name``/``Attribute`` chain, else ``None``."""
    parts: list[str] = []
    node: ast.expr = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    parts.reverse()
    head = aliases.get(parts[0], parts[0])
    return ".".join([head, *parts[1:]])


def _kernel_wrapper(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    """Return the canonical kernel transform that ``expr`` applies, or ``None``.

    Handles ``f``, ``f(...)``, ``partial(f, ...)`` and ``partial(f, ...)(...)`` shapes.
    """
    if isinstance(expr, ast.Call):
        head = resolve_dotted(expr.func, aliases)
        if head == _PARTIAL:
            if not expr.args:
                return None
            target = resolve_dotted(expr.args[0], aliases)
            return target if target in KERNEL_FAMILY else None
        return head if head in KERNEL_FAMILY else None
    name = resolve_dotted(expr, aliases)
    return name if name in KERNEL_FAMILY else None


def is_kernel_decorator(node: ast.expr, aliases: dict[str, str]) -> bool:
    """True when the decorator expression ``node`` applies a JIT- or GRAD-family transform."""
    return _kernel_wrapper(node, aliases) is not None


def _jit_wrapper_of_value(value: ast.expr, aliases: dict[str, str]) -> str | None:
    """Return the JIT-family wrapper that a call-expression value applies to its argument."""
    if not isinstance(value, ast.Call):
        return None
    head = resolve_dotted(value.func, aliases)
    if head in JIT_FAMILY:
        return head
    if isinstance(value.func, ast.Call):
        # partial(jax.jit, ...)(f)
        inner = _kernel_wrapper(value.func, aliases)
        if inner in JIT_FAMILY:
            return inner
    return None


def jitted_name_table(tree: ast.Module, aliases: dict[str, str]) -> dict[str, str]:
    """Map module-local names bound to a JIT-family wrapper onto that wrapper's canonical name."""
    table: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                wrapper = _kernel_wrapper(dec, aliases)
                if wrapper in JIT_FAMILY:
                    table[node.name] = wrapper
                    break
        elif isinstance(node, ast.Assign):
            wrapper = _jit_wrapper_of_value(node.value, aliases)
            if wrapper is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    table[target.id] = wrapper
    return table
