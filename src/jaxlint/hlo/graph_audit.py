"""Semantic JAX kernel auditor using export + StableHLO (optional `jaxlint[hlo]`)."""

from __future__ import annotations

from typing import Any, Callable

import jax
from jax.experimental import checkify
from jax.export import Exported

from jaxlint.hlo.stablehlo_analysis import SemanticAnalyzer


class GraphAuditSensor:
    """StableHLO + checkify-based kernel audit."""

    @staticmethod
    def audit_exported(exported: Exported) -> dict[str, Any]:
        """Run StableHLO violation checks on an already-exported program (no checkify wrap)."""
        analyzer = SemanticAnalyzer(exported)
        analysis = analyzer.parse()

        violations: list[str] = []

        if "stablehlo.while" in analysis.operation_counts:
            violations.append(
                "Implicit 'while' loop detected in StableHLO; prefer jax.lax.scan."
            )

        if "stablehlo.outfeed" in analysis.operation_counts:
            violations.append("Explicit outfeed detected; potential side-effect risk.")

        return {
            "success": len(violations) == 0,
            "violations": violations,
            "operation_counts": analysis.operation_counts,
            "input_types": analysis.type_provenance.get("inputs", []),
        }

    @staticmethod
    def audit_kernel(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        checked_fn = checkify.checkify(fn)
        jit_fn = jax.jit(checked_fn)
        exported = jax.export.export(jit_fn)(*args, **kwargs)
        return GraphAuditSensor.audit_exported(exported)

    @staticmethod
    def instrument_and_run(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Returns checkify(checkify(fn)) result tuple (errors, value)."""
        return checkify.checkify(fn, errors=checkify.all_checks)(*args, **kwargs)
