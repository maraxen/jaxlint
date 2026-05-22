# Sprint artifact: July 2026 — Trust surfaces (approved)

**task_id:** `jaxlint-sprint-may-2026`  
**Plan versions:** v1 NEEDS_WORK (H objective vs code) → v2 **PASS** (plan-auditor 2026-05-05).

## Theme

Strengthen trust signals for **hosted docs** (linkcheck ignores with evidence or dated freeze) and **HLO goldens** (≥2 new `.mlir` fixtures + `needs_jax` tests for paths **other than** the shipped `audit_exported` while/outfeed pair — strategy **B**).

## Task IDs

| ID | Title |
|----|--------|
| D02 | URL / ignore inventory first (`docs/conf.py`); record HTTP outcomes; set **next review date** in `docs/roadmap.md`. |
| D01 | After D02: shrink `linkcheck_ignore` **(i)** with evidence **or (ii)** refresh dated comments + roadmap row. |
| H01 | New `.mlir` + `needs_jax` test — surface A (e.g. `SemanticAnalyzer` inventory on golden text via duck-typed `mlir_module()`). |
| H02 | Second new `.mlir` + test — surface B, orthogonal to H01 (e.g. different op / inventory). |
| H03 | Changelog + roadmap HLO row; fixture **snake_case** names; **churn cap for H01–H03:** ≤6 meaningful paths (2 fixtures + `tests/test_hlo_semantics.py` + `docs/changelog.md` + `docs/roadmap.md` + optional one-line fixture note). |
| R01 | Stretch: ≤90 min contributor blurb — **defer** if H-blockers. |
| V01 | Verification language: `sphinx -W` **HTML only**; linkcheck **no** `-W` (matches `.github/workflows/ci.yml`). |

## Verification

- `uv run ruff check src tests`
- `uv run pytest -q -m smoke`
- `uv run pytest -q -m needs_jax` (with `uv sync --all-groups --extra hlo` + CI-aligned JAX pin when testing HLO)
- `uv run sphinx-build -W -b html docs docs/_build/html`
- `uv run sphinx-build -b linkcheck docs docs/_build/linkcheck` (review output; not `-W`)

## Out of scope

New `audit_exported` op rules without a separate epic; LSP/MCP; `linkcheck` as warnings-as-errors in CI.
