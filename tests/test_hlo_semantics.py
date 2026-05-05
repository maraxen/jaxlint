"""Semantics tests for jaxlint.hlo (requires JAX via jaxlint[hlo])."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from jaxlint.hlo.graph_audit import GraphAuditSensor
from jaxlint.hlo.stablehlo_analysis import SemanticAnalyzer


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_inventory_includes_ops() -> None:
    @jax.jit
    def f(x):
        return x + 1.0

    exp = jax.export.export(f)(jnp.zeros((2, 3), dtype=jnp.float32))
    analyzer = SemanticAnalyzer(exp)
    inv = analyzer.get_operation_inventory()
    assert inv
    assert sum(inv.values()) >= 1
    summary = analyzer.get_summary()
    assert summary["operation_count"] >= 1


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_parse_graph() -> None:
    @jax.jit
    def f(x):
        return x * 2.0

    exp = jax.export.export(f)(jnp.array(0.0, dtype=jnp.float32))
    graph = SemanticAnalyzer(exp).parse()
    assert graph.operation_counts
    assert graph.type_provenance.get("inputs")


@pytest.mark.unit
@pytest.mark.needs_jax
def test_graph_audit_simple_kernel_succeeds() -> None:
    @jax.jit
    def f(x):
        return x + 1.0

    r = GraphAuditSensor.audit_kernel(f, jnp.zeros((1,), dtype=jnp.float32))
    assert r["success"] is True
    assert r["violations"] == []


@pytest.mark.unit
@pytest.mark.needs_jax
def test_graph_audit_while_loop_triggers_violation() -> None:
    @jax.jit
    def f(x):
        def cond(s):
            i, _ = s
            return i < 2

        def body(s):
            i, acc = s
            return i + 1, acc + x

        _, acc = jax.lax.while_loop(cond, body, (0, 0.0))
        return acc

    x = jnp.array(1.0, dtype=jnp.float32)
    exp = jax.export.export(f)(x)
    counts = SemanticAnalyzer(exp).get_operation_inventory()
    assert "stablehlo.while" in counts
    r = GraphAuditSensor.audit_kernel(f, x)
    assert r["success"] is False
    assert r["violations"]
