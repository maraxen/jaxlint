# Plan v5 — HLO diagnostics feasibility DX1 (revised after NEEDS_WORK audit)

**task_id:** `jaxlint-roadmap-remaining-2026`  
**Replaces:** `plan-v4-final-hlo-dx1-2026`  
**Theme:** Phase A metrics → Phase B rules gate.

## Scope recap

**DX1** is a **research + documentation** sprint with optional **Phase B** implementation only if Phase A clears explicit kill criteria.

## Phase A — Spike & metrics (blocking)

### Fixtures **A1–A5** (names illustrative — align with repo)

| ID | Fixture intent | Metric recorded |
|----|----------------|-----------------|
| A1 | Minimal exported stable HLO text snapshot | Parse success, line count |
| A2 | While-loop export path | **`get_operations`** / mapping coverage |
| A3 | Outfeed / side-effect path | Violation detection vs golden |
| A4 | Inventory-style op counts | Count vs mapping drift |
| A5 | Failure / negative control | Explicit **NO-GO** signal |

Fill **Metric recorded** with concrete fields (timestamps optional) before implementation week.

### Kill criteria **K1–K4** (must map to A*)

| Kill ID | Condition | Mapped fixtures |
|---------|-----------|-----------------|
| K1 | Cannot obtain stable text from export API | A1 |
| K2 | Semantic drift across JAX patch versions exceeds agreed threshold | A1–A4 |
| K3 | Parser/heuristic false-positive rate **>** X% on fixtures | A2–A4 |
| K4 | Runtime budget **>** Y ms per fixture on CI hardware | A5 |

Replace **X / Y** with numbers before Phase B gate (oracle approval).

### Deliverables

1. **`docs/adr/0001-hlo-diagnostics-feasibility.md`** (create **`docs/adr`** if missing) — decision **PASS / DEFER / NO-GO** with evidence pointers.
2. Sphinx **toctree** entry under developer/architecture index if present.
3. **`docs/hlo.md`** pointer paragraph → ADR.

## Phase B — Rules (conditional)

- **Prefix:** **`HLO`** + three digits (e.g. **`HLO001`**) — max **5** rules in first drop.
- **CLI:** Prefer **`jaxlint hlo`** subcommand for IR-backed checks if already wired; otherwise **`jaxlint check`** with **`--select HLO*`** — **pick one** in ADR so fixer is not blocked.
- **Blocked until:** Phase A **PASS** row in ADR table.

## Non-goals

- SymPy math, **`shard_map`** UX, LSP embedding of HLO diagnostics (explicit).

## Definition of done (Phase A only for this plan revision)

- [ ] Fixture × metric table committed (this file or **`docs/hlo.md`**).
- [ ] K↔A mapping table committed.
- [ ] ADR skeleton with kill metrics filled or explicitly **TBD** with owner.
- [ ] Roadmap **Medium term** row links ADR when merged.
