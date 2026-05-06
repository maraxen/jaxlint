# Fixer handoff — LSP LP1 documentation

**Plan:** `plan-v4-final-lsp-docs-2026` (plan-audit **PASS**)  
**task_id:** `jaxlint-roadmap-remaining-2026`

## Source of truth

Production server reads **only** **`JAXLINT_LSP_DEBOUNCE_MS`** (`src/jaxlint/lsp/server.py` — **`_lsp_debounce_ms()`**). Tests may set **`50`** / **`0`** for timing (`tests/test_lsp_stdio.py`). Do **not** invent extra **`JAXLINT_*`** vars without code changes.

## Tasks

### LP1 — Structure

- Refactor **`docs/lsp.md`** **Behaviour** section: add **Environment & lifecycle** with:
  - Debounce (**env**: **`JAXLINT_LSP_DEBOUNCE_MS`**).
  - **Non-env lifecycle** rows (same table or companion bullets): **`didOpen`** immediate; **`didChange`** debounced; **`didClose`** clears diagnostics; **`file://`** only.

### LP1 — Env table (minimum columns: Variable | Meaning | Default)

| Variable | Meaning | Default |
|----------|---------|---------|
| **`JAXLINT_LSP_DEBOUNCE_MS`** | Milliseconds to debounce **`textDocument/didChange`** before lint | **200** (**`0`** = immediate sequential path) |

Add footnote: *No other **`JAXLINT_*`** variables are read by the server in **0.4.0**.*

### LP1 — Cross-links

- **`docs/contributing.md`** → short bullet under testing: run **`pytest -m lsp`** when touching **`jaxlint/lsp/`** (match CI).

## DoD

- **`sphinx-build -W`** clean for edited docs (local or CI parity).
- Optional: **`Diagnostic`** / **`tags`** — **out of scope** unless **`Diagnostic`** gains fields (per earlier LP plan).
