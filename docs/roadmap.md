# Roadmap

This page is the **canonical** roadmap. The README carries a short pointer so it does not drift from here.

## Product vision

**jaxlint** stays **easy to drive from CI and editors**: a JAX-aware static checker and doc-quality tool whose **library surface** (`jaxlint.core`) stays **import-light** (no JAX) while **`jaxlint[hlo]`** can go deeper where JAX and exports are already available.

## Shipped baseline (approximately v0.1)

Already in tree today:

| Area | Status |
|------|--------|
| **CLI** (`check`, `doc`, `score`, `rules`, `init`, `version`, `hlo`) | Stable for core flows; `check`/`doc` support `--format text`, `compact`, `json`, `github` / `github-annotations`, or **SARIF** 2.1.0 (`sarif` / `sarif-2.1.0`). |
| **Rules** | JL/JD/JM catalogue with fixtures and docs. |
| **Config** | `[tool.jaxlint]` parsing, selects/ignores/per-file ignores. |
| **Testing** | Pytest tiers (`smoke` / `unit` / `integration` / `needs_jax`), coverage floor, GitHub Actions. Pull requests run **smoke** + lint; **full pytest + coverage** runs on **`main` pushes only** (`lint-test`). |
| **Docs** | Sphinx + MyST + Read the Docs (HTML plus **PDF** plus **htmlzip** on version builds). |
| **`jaxlint[hlo]`** | Optional imports; CLI only checks the extra — **merge-blocking `needs_jax` tests** cover export → `SemanticAnalyzer` / `GraphAuditSensor` behaviour (including `audit_exported`, while/outfeed rules, `get_operations`, checkify edges), not only import smoke. |

## Near term (next few releases)

Focus: **integrations, ergonomics, and trust** without bloating `jaxlint.core`.

1. **Output formats** — **Shipped:** GitHub annotation JSON, **SARIF** 2.1.0 (`--format sarif`), and **`compact`** text. **Remaining:** consider **Rich**-based human layout (Typer already pulls Rich transitively).
2. **Hosted docs hygiene** — **`sphinx.fail_on_warning` is already on** in `.readthedocs.yml`. **Remaining:** shrink **`linkcheck_ignore`** in `docs/conf.py` once the canonical RTD/HTML URLs and GitHub targets respond reliably to automated linkcheck (the root Read the Docs URL and GitHub may 404 or rate-limit until the project and links are stable).
3. **CI parity** — **Shipped:** required **`jax-hlo`** job installs `jaxlint[hlo]` and runs `pytest -m needs_jax` (merge-blocking). PR **`lint-test`** jobs run smoke + Ruff; full suite + coverage on **`main`** only.
4. **API reference** — **Shipped:** Sphinx **autodoc** / **autosummary** for `jaxlint.core` (and configuration types under `jaxlint.core.config`; see **`api`** in the docs toctree). **`jaxlint.hlo`** stays a narrative-only page (`hlo`) so JAX is not imported on RTD builders.
5. **HLO credibility** — **Shipped:** targeted pytest for `SemanticAnalyzer` / `GraphAuditSensor` (`get_operations`, `audit_kernel` / `audit_exported`, while and outfeed violation paths, `instrument_and_run` checkify failure). **Remaining:** optional **golden IR** snippets (or broader MLIR parsing) if we want version-pinned IR regression tests without relying only on live `jax.export` text.

## Medium term

| Initiative | Goal |
|------------|------|
| **Language Server (LSP)** | Thin wrapper that streams **`Diagnostic`** objects to editors (diagnostics-only v1; formatting out of scope). |
| **HLO diagnostics** | Optional analysers over lowered IR/text (similar in spirit to “HLO dump regex” tooling described in jaxbeans discussions): scatter→gather fusion gaps, noisy kernel launches — scoped to **`jaxlint.hlo`**. |
| **Performance at scale** | Caching incremental results per file, parallelism for repo-wide runs. |
| **Config evolution** | Per-rule overrides, SARIF/GitHub severity mapping, reproducible **`jaxlint.toml`** aside from `[tool.jaxlint]`. |

## Longer horizon / exploratory

These are **not committed** milestones; revisit when upstream JAX and IDE ecosystems move.

| Idea | Notes |
|------|--------|
| **MCP server** | Separate package or subpackage exposing the same primitives as MCP tools/resources for agent workflows. |
| **Math semantics** | Beyond delimiter/LaTeX structure: optional SymPy-backed checks — opt-in dependency. |
| **Deeper parallelism / sharding UX** | New rules aligned with JAX `jit`/`shard_map` ergonomics as patterns stabilize. |

## How to steer this roadmap

Open issues or RFC-style PRs that spell out **rule IDs**, **default severity**, **config keys**, and **test fixtures**. Breaking changes should bump minor/major semver and mention migration in Sphinx **changelog** once that doc exists.
