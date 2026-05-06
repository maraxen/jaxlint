"""JL001 scatter+gather, JL002 short @jax.jit helper."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.diagnostic import Diagnostic, Severity


def scatter_gather_diagnostics(tree: ast.Module, path: Path) -> list[Diagnostic]:
    warnings: list[Diagnostic] = []
    sg = ScatterGatherVisitor(path)
    sg.visit(tree)
    warnings.extend(sg.diagnostics)
    return warnings


class ScatterGatherVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.diagnostics: list[Diagnostic] = []

    def _is_at_set(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set"
            and isinstance(node.func.value, ast.Subscript)
            and isinstance(node.func.value.value, ast.Attribute)
            and node.func.value.value.attr == "at"
        )

    def _at_set_base(self, node: ast.expr) -> str | None:
        """Recover base name from ``x.at[...].set(...)`` (aligned with `_is_at_set`)."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "set":
            return None
        sub = func.value
        if not isinstance(sub, ast.Subscript):
            return None
        at_chain = sub.value
        if not isinstance(at_chain, ast.Attribute) or at_chain.attr != "at":
            return None
        base = at_chain.value
        if isinstance(base, ast.Name):
            return base.id
        return None

    def visit_Module(self, node: ast.Module) -> None:
        self._tag_parent_bodies(node)
        self.generic_visit(node)

    def _tag_parent_bodies(self, node: ast.AST) -> None:
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.AST):
                for child in value:
                    child._parent_body = value  # type: ignore[attr-defined]
                    self._tag_parent_bodies(child)
            elif isinstance(value, ast.AST):
                self._tag_parent_bodies(value)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_at_set(node.value):
            base = self._at_set_base(node.value)
            lhs_names: set[str] = set()
            for t in node.targets:
                if isinstance(t, ast.Name):
                    lhs_names.add(t.id)

            parent_body = getattr(node, "_parent_body", None)
            if parent_body is not None:
                idx = next((i for i, n in enumerate(parent_body) if n is node), None)
                if idx is not None:
                    for later in parent_body[idx + 1 :]:
                        for child in ast.walk(later):
                            if (
                                isinstance(child, ast.Subscript)
                                and isinstance(child.value, ast.Name)
                                and child.value.id in lhs_names
                            ):
                                msg = (
                                    f"`.at[].set()` on '{base}' immediately consumed via "
                                    f"`{child.value.id}[...]`. Prefer gather + jnp.where."
                                )
                                self.diagnostics.append(
                                    Diagnostic("JL001", Severity.ERROR, msg, self.path, node.lineno)
                                )
                                break
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        has_jit = any(self._decorator_is_jit(d) for d in node.decorator_list)
        if has_jit:
            body_stmts = [n for n in ast.walk(node) if isinstance(n, ast.stmt)]
            if len(body_stmts) < 15:
                self.diagnostics.append(
                    Diagnostic(
                        "JL002",
                        Severity.WARNING,
                        f"`{node.name}` has `@jax.jit` and looks like a short helper — "
                        "may prevent XLA fusion; remove JIT from helpers.",
                        self.path,
                        node.lineno,
                    )
                )
        self.generic_visit(node)

    @staticmethod
    def _decorator_is_jit(d: ast.AST) -> bool:
        if isinstance(d, ast.Attribute) and d.attr == "jit":
            return True
        return (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "jit"
        )
