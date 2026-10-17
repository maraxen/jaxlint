"""AST-based performance sensors (no JAX runtime)."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxlint.core.config import LintConfig
from jaxlint.core.diagnostic import Diagnostic
from jaxlint.core.sensors.best_practices import best_practices_diagnostics
from jaxlint.core.sensors.fresh_jit import fresh_jit_diagnostics
from jaxlint.core.sensors.loops import loop_diagnostics
from jaxlint.core.sensors.purity import purity_diagnostics
from jaxlint.core.sensors.scatter_gather import scatter_gather_diagnostics
from jaxlint.core.sensors.shapes import shape_diagnostics


def perf_diagnostics_from_tree(path: Path, cfg: LintConfig, tree: ast.Module) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(scatter_gather_diagnostics(tree, path))
    diagnostics.extend(loop_diagnostics(tree, path))
    diagnostics.extend(purity_diagnostics(tree, path))
    diagnostics.extend(shape_diagnostics(tree, path, strict=cfg.strict_shapes))
    diagnostics.extend(best_practices_diagnostics(tree, path))
    diagnostics.extend(fresh_jit_diagnostics(tree, path))
    return diagnostics


def perf_diagnostics_for_file(path: Path, cfg: LintConfig) -> list[Diagnostic]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    return perf_diagnostics_from_tree(path, cfg, tree)
