"""JL021: host-side loop over replicates/resamples whose replicate axis should be batched."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity
from jaxlint.core.sensors._jax_names import alias_map, is_kernel_decorator

_REPLICATE_RE = re.compile(
    r"boot|resampl|replicat|n_?perm|permutation|n_?draw|n_?rep\b|n_?reps"
    r"|n_?trial|n_?seed|n_?sample",
    re.IGNORECASE,
)
_TRIVIAL_NAMES: frozenset[str] = frozenset(
    {
        "print",
        "len",
        "range",
        "str",
        "format",
        "tqdm",
        "float",
        "int",
        "bool",
        "tuple",
        "list",
        "dict",
        "set",
        "zip",
        "enumerate",
        "isinstance",
        "sorted",
        "min",
        "max",
        "abs",
        "sum",
    }
)
_TRIVIAL_ATTRS: frozenset[str] = frozenset({"append", "extend", "format"})
_TRIVIAL_CHAIN_ROOTS: frozenset[str] = frozenset({"logging", "logger", "time", "tqdm"})


def replicate_loop_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    visitor = ReplicateLoopSensor(path)
    visitor.visit(tree)
    return visitor.diagnostics


class ReplicateLoopSensor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.diagnostics: list[Diagnostic] = []
        self.aliases: dict[str, str] = {}
        self._kernel_depth = 0
        self._flagged_names: list[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.aliases = alias_map(node)
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_kernel = any(is_kernel_decorator(d, self.aliases) for d in node.decorator_list)
        self._kernel_depth += int(is_kernel)
        self.generic_visit(node)
        self._kernel_depth -= int(is_kernel)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_For(self, node: ast.For) -> None:
        name = _replicate_name(node.iter)
        flagged = False
        if name is not None and self._kernel_depth == 0 and name not in self._flagged_names:
            call = _first_nontrivial_call(node.body)
            if call is not None and _accumulates(node.body):
                flagged = True
                callee = ast.unparse(call.func)[:60]
                self.diagnostics.append(
                    Diagnostic(
                        "JL021",
                        Severity.INFO,
                        f"Host loop over {name} replicates calls {callee} per iteration; "
                        "batch the replicate axis (jax.vmap / batched NumPy) and measure "
                        "with `xtrax profile`.",
                        self.path,
                        node.lineno,
                    )
                )
        if flagged and name is not None:
            self._flagged_names.append(name)
        self.generic_visit(node)
        if flagged:
            self._flagged_names.pop()


def _identifier_matches(expr: ast.expr) -> str | None:
    """Return the first Name/Attribute identifier under ``expr`` matching the replicate pattern."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            ident = node.id
        elif isinstance(node, ast.Attribute):
            ident = node.attr
        else:
            continue
        if _REPLICATE_RE.search(ident):
            return ident
    return None


def _replicate_name(iterable: ast.expr) -> str | None:
    """Replicate identifier when ``iterable`` is ``range(..)``, ``enumerate(..)`` or a bare name."""
    if isinstance(iterable, ast.Call):
        func = iterable.func
        if isinstance(func, ast.Name) and func.id == "range" and iterable.args:
            return _identifier_matches(iterable.args[-1])
        if isinstance(func, ast.Name) and func.id == "enumerate" and iterable.args:
            return _replicate_name(iterable.args[0])
        return None
    if isinstance(iterable, (ast.Name, ast.Attribute)):
        return _identifier_matches(iterable)
    return None


def _chain_parts(expr: ast.expr) -> list[str]:
    parts: list[str] = []
    node: ast.expr = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts


def _is_trivial_callee(func: ast.expr) -> bool:
    parts = _chain_parts(func)
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in _TRIVIAL_NAMES
    if parts[0] in _TRIVIAL_ATTRS:
        return True
    return any(p in _TRIVIAL_CHAIN_ROOTS for p in parts)


def _first_nontrivial_call(body: list[ast.stmt]) -> ast.Call | None:
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and not _is_trivial_callee(node.func):
                return node
    return None


def _accumulates(body: list[ast.stmt]) -> bool:
    """True when the body appends/extends, augments with a call result, or subscript-assigns."""
    for stmt in body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("append", "extend")
            ):
                return True
            if isinstance(node, ast.AugAssign) and any(
                isinstance(n, ast.Call) for n in ast.walk(node.value)
            ):
                return True
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(n, ast.Subscript) for t in targets for n in ast.walk(t)):
                    return True
    return False
