# Plan v5 — HLO goldens & credibility (revised after NEEDS_WORK audit)

**task_id:** `jaxlint-roadmap-remaining-2026`  
**Replaces:** `plan-v4-final-hlo-goldens-2026`  
**Theme:** Golden semantics, timeouts, docs links.

## Problem statement

Auditors flagged missing acceptance criteria for **source-mapping semantics vs operation-count aggregates**, and ambiguous **pytest-timeout** scope for expensive **`needs_jax`** tests.

## Acceptance criteria (mandatory)

### HG1 — Mapping semantics (documentation + tests)

- **Canonical rule:** When documenting `get_source_mapping` / golden helpers, state explicitly whether map keys are **last-write-wins** per `(kind, line)` or merged — match **implementation** (grep `get_source_mapping`, golden loaders under `tests/fixtures/hlo/`).
- **Contrast:** **`operation_counts`** (or equivalent counters) **increment** per observation; goldens that snapshot counts must not contradict mapping snapshots unless the doc explains why both views coexist.
- **Deliverable:** One subsection in **`docs/hlo.md`** (or contributing) titled **Goldens: mapping vs counts** with a bullet table.

### HG2 — pytest-timeout scope

- **Scope:** Apply **`pytest-timeout`** (or **`@pytest.mark.timeout(60)`**) to **`needs_jax`** tests that run lowered/HLO paths — **not** the entire repo unless already configured.
- **Concrete:** Add **`timeout = 60`** (or project-standard seconds) via **`pytest.ini`** / **`pyproject.toml`** **`[tool.pytest.ini_options]`** using **`timeout_func_only = true`** if using `pytest-timeout`, **or** mark only `tests/` modules under the HLO golden suite that carry **`needs_jax`**.
- **Escape hatch:** Long-running tests document **`timeout`** override in module docstring.

### HG3 — Contributing canonical path

- **`docs/contributing.md`** → link **Run HLO / golden tests** → `tests/fixtures/hlo/README` if present, else **`docs/hlo.md`** § goldens.

### HG4 — CI / docs alignment

- **`jax-hlo`** job unchanged unless timeout causes flakes — then bump threshold with justification in changelog snippet.

## Non-goals

- Full MLIR/XLA dump parsers.
- Replacing duck-typed fixtures with real JAX exports unless a separate spike says otherwise.

## Definition of done

- [ ] Mapping vs counts subsection merged.
- [ ] Timeout configured for **`needs_jax`** HLO golden paths; CI green.
- [ ] Contributing cross-link present.
- [ ] No regression in existing golden assertions.

## Estimated packages

| ID | Deliverable |
|----|----------------|
| HG1 | Doc + test comments aligning mapping/count semantics |
| HG2 | Timeout wiring + rationale in commit |
| HG3 | Contributing link |
| HG4 | CI observation note if job timing changes |
