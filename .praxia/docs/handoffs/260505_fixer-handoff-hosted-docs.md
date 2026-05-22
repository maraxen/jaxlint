# Fixer handoff — Hosted docs HD1–HD4

**Plan:** `plan-v4-final-hosted-docs-2026` (plan-audit **PASS**)  
**task_id:** `jaxlint-roadmap-remaining-2026`

## Goal

Reduce canonical URL drift and make CI/docs behaviour obvious for contributors.

## Tasks

### HD1 — CI path filtering

- Ensure workflows run **docs builds** on **`main`** always (full Sphinx or RTD-equivalent).
- On PRs, run **path-filtered** docs job when **`docs/**`**, **`docs/conf.py`**, **`.readthedocs.yml`**, **`README.md`** (if it embeds URLs), or **`docs/roadmap.md`** change — align with existing workflow names in `.github/workflows/`.
- **Deliverable:** Comment at top of workflow listing **glob triggers** (audit nit: frozen globs).

### HD2 — Canonical URL table

- In **`docs/roadmap.md`** or dedicated **`docs/hosting.md`** (avoid duplication): table columns — **URL**, **Kind** (RTD project, GitHub repo, branch docs), **Last checked** (date), **HTTP status** (404/200), **`linkcheck_ignore` rationale** if ignored, **Next review** (**2026-10-01** per roadmap).

### HD3 — Contributing merge

- **`docs/contributing.md`**: single subsection **Hosted documentation & links** pointing to canonical table + **`docs/conf.py`** **`linkcheck_ignore`** block comments.

### HD4 — Warnings vs linkcheck

- One short paragraph: **`sphinx-build -W`** / RTD **`fail_on_warning: true`** treats Sphinx warnings as failure; **`linkcheck`** may run in a separate job — clarify whether **`make linkcheck`** uses **`-W`** in CI (match reality).

## DoD

- `sphinx-build` / CI docs job succeeds on branch.
- Changelog entry if user-visible.
- Roadmap Near term §2 updated only if facts changed.

## Non-goals

- Claiming HTTP **200** for URLs that still **404** — record truth.
