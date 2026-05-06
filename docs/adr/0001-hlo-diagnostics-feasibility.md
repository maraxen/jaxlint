# ADR 0001 — HLO diagnostics feasibility (Phase A)

**Status:** Draft — Phase A metrics  
**Date:** 2026-05-05  
**Context:** [Roadmap — Medium term](../roadmap.md), optional **`jaxlint.hlo`** analysers over StableHLO text.

## Decision (pending)

Phase B (**`HLO*`** rules, ≤5 in first drop) is **blocked** until Phase A metrics below are filled and kill criteria reviewed (**oracle**). Default CLI surface for IR-backed checks: **`jaxlint check`** with **`--select HLO*`**; the **`jaxlint hlo`** command remains an **import/extra smoke** hint until extended deliberately.

## Fixture × metric matrix

Maps spike fixtures under `tests/fixtures/hlo/` to Phase A metrics. IDs align with planning docs (A1–A5).

| ID | Fixture file | Metric recorded |
|----|----------------|-----------------|
| **A1** | `add_constant_kernel.mlir` | Minimal StableHLO text parse success; **operation_inventory** keys present |
| **A2** | `while_loop_detected.mlir` | **`stablehlo.while`** path; **`get_operations`** / mapping coverage |
| **A3** | `outfeed_side_effect.mlir` | Side-effect / audit_exported violation detection vs golden |
| **A4** | `multiply_reshape_kernel.mlir` | Inventory counts vs **`get_source_mapping`** drift |
| **A5** | *(CI harness / negative path — TBD)* | Explicit failure or timeout budget signal for **NO-GO** |

## Kill criteria (K1–K4)

| ID | Condition | Mapped fixtures | Threshold |
|----|-----------|-----------------|-----------|
| **K1** | Cannot obtain stable MLIR text from supported export API | A1 | *(binary)* |
| **K2** | Semantic drift across JAX patch versions | A1–A4 | **TBD** — owner assigns max allowed diff |
| **K3** | Parser/heuristic false-positive rate on fixtures | A2–A4 | **TBD** — max **X** % |
| **K4** | Wall-clock per fixture on CI hardware | A5 | **TBD** — max **Y** ms |

**X / Y** require maintainer sign-off before Phase B.

## Phase B preview (not active)

- Rule IDs: **`HLO`** + three digits, max **5** rules initially.
- Non-goals: SymPy-backed math, **`shard_map`** UX, embedding HLO diagnostics in LSP (see [Language server](../lsp.md)).

## Links

- Narrative install / import story: [HLO extra](../hlo.md)
- Golden semantics: [Goldens: mapping vs counts](../hlo.md#goldens-mapping-vs-counts)
