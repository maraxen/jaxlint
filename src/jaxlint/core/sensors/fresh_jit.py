"""JL022: a JIT-family wrap of a fresh callable inside a host loop (retraces every iteration).

Re-wrapping the same function object inside a loop hits the jit cache. Wrapping a callable that
is created anew each iteration (a lambda, a ``functools.partial``, a ``def`` inside the loop body)
retraces and recompiles every time. The rule fires only when the fresh callable depends on
nothing loop-varying: hoisting the wrap out of the loop is then a pure win. A callable whose free
names include the loop target, or a name assigned in the loop body, is left alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.sensors._jax_names import JIT_FAMILY, alias_map, resolve_dotted

_PARTIAL = "functools.partial"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def fresh_jit_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    visitor = FreshJitSensor(path, alias_map(tree))
    visitor.visit(tree)
    return visitor.diagnostics


class FreshJitSensor(ast.NodeVisitor):
    def __init__(self, path: Path, aliases: dict[str, str]):
        self.path = path
        self.aliases = aliases
        self.diagnostics: list[Diagnostic] = []
        self._loops: list[ast.For | ast.AsyncFor | ast.While] = []

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        self._loops.append(node)
        self.generic_visit(node)
        self._loops.pop()

    def _visit_scope(self, node: ast.AST) -> None:
        # A function body runs when called, not once per iteration of an enclosing loop.
        saved, self._loops = self._loops, []
        self.generic_visit(node)
        self._loops = saved

    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope

    def visit_Call(self, node: ast.Call) -> None:
        if self._loops:
            self._check_wrap(node, self._loops[-1])
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._loops:
            for dec in node.decorator_list:
                wrapper = _jit_decorator_wrapper(dec, self.aliases)
                if wrapper is not None:
                    self._report_if_independent(node, self._loops[-1], dec, wrapper)
                    break
        self._visit_scope(node)

    # -- checks -----------------------------------------------------------------

    def _check_wrap(self, call: ast.Call, loop: ast.For | ast.AsyncFor | ast.While) -> None:
        wrapper = _jit_wrapper_of_call(call, self.aliases)
        if wrapper is None:
            return
        fn_arg = _callable_argument(call)
        if fn_arg is None:
            return
        fresh = _fresh_callable(fn_arg, loop, self.aliases)
        if fresh is not None:
            self._report_if_independent(fresh, loop, call, wrapper)

    def _report_if_independent(
        self,
        fresh: ast.AST,
        loop: ast.For | ast.AsyncFor | ast.While,
        at: ast.expr | ast.stmt,
        wrapper: str,
    ) -> None:
        if _free_names(fresh) & _loop_varying_names(loop):
            return
        self.diagnostics.append(_diagnostic(self.path, at, wrapper))


def _jit_decorator_wrapper(dec: ast.expr, aliases: dict[str, str]) -> str | None:
    """Canonical JIT-family name for a decorator: ``@jax.jit``, ``@jax.jit(...)``, ``@partial(jax.jit, ...)``."""
    if isinstance(dec, ast.Call):
        head = resolve_dotted(dec.func, aliases)
        if head == _PARTIAL and dec.args:
            head = resolve_dotted(dec.args[0], aliases)
    else:
        head = resolve_dotted(dec, aliases)
    return head if head in JIT_FAMILY else None


def _diagnostic(path: Path, node: ast.expr | ast.stmt, wrapper: str) -> Diagnostic:
    return Diagnostic(
        "JL022",
        Severity.WARNING,
        f"`{wrapper}` wraps a new callable every iteration and retraces each time; hoist the wrap "
        "out of the loop (the callable does not depend on loop variables).",
        path,
        node.lineno,
        node.col_offset,
    )


def _jit_wrapper_of_call(call: ast.Call, aliases: dict[str, str]) -> str | None:
    """Canonical JIT-family name applied by ``call``: ``jax.jit(f)`` or ``partial(jax.jit, ...)(f)``."""
    head = resolve_dotted(call.func, aliases)
    if head in JIT_FAMILY:
        return head
    if isinstance(call.func, ast.Call) and resolve_dotted(call.func.func, aliases) == _PARTIAL:
        if call.func.args:
            target = resolve_dotted(call.func.args[0], aliases)
            if target in JIT_FAMILY:
                return target
    return None


def _callable_argument(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "fun":
            return kw.value
    return None


def _fresh_callable(
    arg: ast.expr, loop: ast.For | ast.AsyncFor | ast.While, aliases: dict[str, str]
) -> ast.AST | None:
    """Return the node that defines a callable created anew each iteration, else ``None``."""
    if isinstance(arg, ast.Lambda):
        return arg
    if isinstance(arg, ast.Call) and resolve_dotted(arg.func, aliases) == _PARTIAL:
        return arg
    if isinstance(arg, ast.Name):
        return _loop_body_definition(arg.id, loop, aliases)
    return None


def _loop_body_definition(name: str, loop: ast.AST, aliases: dict[str, str]) -> ast.AST | None:
    """A ``def``, lambda-assignment or partial-assignment binding ``name`` inside the loop body."""
    for node in _walk_loop_body(loop):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Lambda):
                return value
            if isinstance(value, ast.Call) and resolve_dotted(value.func, aliases) == _PARTIAL:
                return value
    return None


def _walk_loop_body(loop: ast.AST):
    """Nodes of the loop body, not descending into nested function scopes."""
    stack: list[ast.AST] = list(getattr(loop, "body", []))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(node.decorator_list)
            continue
        stack.extend(ast.iter_child_nodes(node))


def _loop_varying_names(loop: ast.For | ast.AsyncFor | ast.While) -> set[str]:
    """Loop target(s) plus every name stored anywhere in the loop body.

    A name stored later in the body is still loop-varying: its value from the previous
    iteration reaches the wrap, so counting it keeps the rule conservative.
    """
    names: set[str] = set()
    if isinstance(loop, (ast.For, ast.AsyncFor)):
        names |= _stored_names(loop.target)
    for node in _walk_loop_body(loop):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _stored_names(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


def _param_names(args: ast.arguments) -> set[str]:
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        params.append(args.vararg)
    if args.kwarg:
        params.append(args.kwarg)
    return {p.arg for p in params}


def _default_names(args: ast.arguments) -> set[str]:
    """Names read by default-value expressions (evaluated in the enclosing scope)."""
    exprs = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
    return {n.id for e in exprs for n in ast.walk(e) if isinstance(n, ast.Name)}


def _free_names(node: ast.AST) -> set[str]:
    """Names a fresh callable reads from its enclosing scope (approximate, ast-only)."""
    if isinstance(node, ast.Lambda):
        loaded = {n.id for n in ast.walk(node.body) if isinstance(n, ast.Name)}
        return (loaded - _param_names(node.args)) | _default_names(node.args)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        local = _param_names(node.args)
        loaded: set[str] = set()
        for stmt in node.body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name):
                    if isinstance(n.ctx, ast.Load):
                        loaded.add(n.id)
                    else:
                        local.add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    local.add(n.name)
        # Defaults are evaluated when the def runs, so they bind the loop value at definition;
        # a default may share a name with a parameter (``nl=nl``), so they are added after the subtraction.
        return (loaded - local) | _default_names(node.args)
    # functools.partial(f, ...) call: everything it reads is in its arguments
    loaded = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loaded.add(n.id)
    return loaded
