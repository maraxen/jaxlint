"""Semantics tests for jaxlint.hlo (requires JAX via jaxlint[hlo]).

Golden semantics (**inventory vs mapping**) are documented under
`Goldens: mapping vs counts` in ``docs/hlo.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from jaxlint.hlo.graph_audit import GraphAuditSensor
from jaxlint.hlo.stablehlo_analysis import SemanticAnalyzer

_HLO_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hlo"


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
    assert "stablehlo.add" in r["operation_counts"]
    assert len(r["input_types"]) == 1
    inp = r["input_types"][0]
    assert "dtype" in inp and "shape" in inp


@pytest.mark.unit
@pytest.mark.needs_jax
def test_graph_audit_exported_outfeed_violation_fake() -> None:
    """Duck-typed export: mlir mentions outfeed → one side-effect violation, success False."""

    mlir_text = (_HLO_FIXTURES / "outfeed_side_effect.mlir").read_text(encoding="utf-8")

    class FakeExported:
        def mlir_module(self) -> str:
            return mlir_text

        in_avals = ()
        out_avals = ()

    r = GraphAuditSensor.audit_exported(FakeExported())  # type: ignore[arg-type]
    assert r["success"] is False
    assert len(r["violations"]) == 1
    v = r["violations"][0]
    assert "outfeed" in v.lower()
    assert "side-effect" in v.lower() or "side effect" in v.lower()


@pytest.mark.unit
@pytest.mark.needs_jax
def test_graph_audit_exported_while_violation_fake() -> None:
    """Golden MLIR: ``stablehlo.while`` in text → inventory + ``audit_exported`` while violation (no live export)."""

    mlir_text = (_HLO_FIXTURES / "while_loop_detected.mlir").read_text(encoding="utf-8")
    assert "stablehlo.while" in mlir_text

    class FakeExported:
        def mlir_module(self) -> str:
            return mlir_text

        in_avals = ()
        out_avals = ()

    r = GraphAuditSensor.audit_exported(FakeExported())  # type: ignore[arg-type]
    assert r["success"] is False
    assert len(r["violations"]) == 1
    msg = "Implicit 'while' loop detected in StableHLO; prefer jax.lax.scan."
    assert r["violations"] == [msg]
    assert r["operation_counts"].get("stablehlo.while", 0) >= 1


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_inventory_golden_add_constants_fake_export() -> None:
    """Golden MLIR: constants + ``stablehlo.add`` — inventory / ``get_operations`` (no ``audit_exported``)."""

    mlir_text = (_HLO_FIXTURES / "add_constant_kernel.mlir").read_text(encoding="utf-8")

    class FakeExported:
        def mlir_module(self) -> str:
            return mlir_text

        in_avals = ()
        out_avals = ()

    analyzer = SemanticAnalyzer(FakeExported())  # type: ignore[arg-type]
    inv = analyzer.get_operation_inventory()
    assert inv.get("stablehlo.constant", 0) >= 2
    assert inv.get("stablehlo.add", 0) >= 1
    full_names = {op.full_name for op in analyzer.get_operations()}
    assert {"stablehlo.constant", "stablehlo.add"}.issubset(full_names)
    summary = analyzer.get_summary()
    assert summary["operation_count"] >= 3


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_inventory_golden_multiply_reshape_fake_export() -> None:
    """Golden MLIR: ``stablehlo.multiply`` + ``stablehlo.reshape`` + constants — distinct inventory path."""

    mlir_text = (_HLO_FIXTURES / "multiply_reshape_kernel.mlir").read_text(encoding="utf-8")

    class FakeExported:
        def mlir_module(self) -> str:
            return mlir_text

        in_avals = ()
        out_avals = ()

    analyzer = SemanticAnalyzer(FakeExported())  # type: ignore[arg-type]
    inv = analyzer.get_operation_inventory()
    assert inv.get("stablehlo.multiply", 0) >= 1
    assert inv.get("stablehlo.reshape", 0) >= 1
    assert inv.get("stablehlo.constant", 0) >= 2
    full_names = {op.full_name for op in analyzer.get_operations()}
    assert "stablehlo.multiply" in full_names
    assert "stablehlo.reshape" in full_names


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_get_operations_add_kernel() -> None:
    @jax.jit
    def f(x):
        return x + 1.0

    exp = jax.export.export(f)(jnp.zeros((2, 3), dtype=jnp.float32))
    full_names = {op.full_name for op in SemanticAnalyzer(exp).get_operations()}
    assert {"stablehlo.add", "stablehlo.broadcast_in_dim"}.issubset(full_names)


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_get_operations_multiply_kernel() -> None:
    @jax.jit
    def f(x, y):
        return x * y

    exp = jax.export.export(f)(
        jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
    )
    full_names = {op.full_name for op in SemanticAnalyzer(exp).get_operations()}
    assert "stablehlo.multiply" in full_names


@pytest.mark.unit
@pytest.mark.needs_jax
def test_semantic_analyzer_get_operations_while_includes_while_op() -> None:
    @jax.jit
    def g(x):
        def cond(s):
            i, _ = s
            return i < 2

        def body(s):
            i, acc = s
            return i + 1, acc + x

        _, acc = jax.lax.while_loop(cond, body, (0, 0.0))
        return acc

    exp = jax.export.export(g)(jnp.array(1.0, dtype=jnp.float32))
    ops = SemanticAnalyzer(exp).get_operations()
    assert any(op.full_name == "stablehlo.while" for op in ops)


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
    assert r["operation_counts"].get("stablehlo.while", 0) >= 1
    msg = "Implicit 'while' loop detected in StableHLO; prefer jax.lax.scan."
    assert r["violations"] == [msg]


@pytest.mark.unit
@pytest.mark.needs_jax
def test_graph_audit_instrument_and_run_checkify_path() -> None:
    """Covers ``instrument_and_run`` (checkify wrapper); distinct from ``audit_kernel`` export plumbing."""

    def f(x):
        return x + 1.0

    err, val = GraphAuditSensor.instrument_and_run(f, jnp.array(1.0, dtype=jnp.float32))
    assert float(val) == pytest.approx(2.0)
    err.throw()


@pytest.mark.unit
@pytest.mark.needs_jax
def test_graph_audit_instrument_and_run_checkify_reports_failure() -> None:
    def f(x):
        checkify.check(x > 0, "must be positive")
        return x

    err, val = GraphAuditSensor.instrument_and_run(f, jnp.array(-1.0, dtype=jnp.float32))
    assert float(val) == pytest.approx(-1.0)
    err_str = str(err)
    assert "must be positive" in err_str
    assert "check" in err_str.lower() or "failed" in err_str.lower()
    with pytest.raises(Exception, match="must be positive"):
        err.throw()
