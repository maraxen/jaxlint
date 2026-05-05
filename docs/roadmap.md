# Roadmap

This page is the **canonical** roadmap. The README carries a short pointer so it does not drift from here.

## Product vision

**jaxlint** stays **easy to drive from CI and editors**: a JAX-aware static checker and doc-quality tool whose **library surface** (`jaxlint.core`) stays **import-light** (no JAX) while **`jaxlint[hlo]`** can go deeper where JAX and exports are already available.

## Shipped baseline (aligned with 0.2.x)

This section tracks what is **in the repository today**. **Packaged releases** use the `version` in `pyproject.toml` (currently **0.2.0**); treat that as the source of truth for semver, not an informal “v0.1” label.

Already in tree today:

| Area | Status |
|------|--------|
| **CLI** (`check`, `doc`, `score`, `rules`, `init`, `version`, `hlo`) | Stable for core flows; `check`/`doc` support `--format text`, `compact`, `json`, `github` / `github-annotations`, **SARIF** 2.1.0 (`sarif` / `sarif-2.1.0`), or **`rich`** (table layout). |
| **Rules** | JL/JD/JM catalogue with fixtures and docs. |
| **Config** | `[tool.jaxlint]` parsing, selects/ignores/per-file ignores. |
| **Testing** | Pytest tiers (`smoke` / `unit` / `integration` / `needs_jax`), coverage floor, GitHub Actions. Pull requests run **smoke** + lint; **full pytest + coverage** runs on **`main` pushes only** (`lint-test`). |
| **Docs** | Sphinx + MyST + Read the Docs (HTML plus **PDF** plus **htmlzip** on version builds). |
| **`jaxlint[hlo]`** | Optional imports; CLI only checks the extra — **merge-blocking `needs_jax` tests** cover export → `SemanticAnalyzer` / `GraphAuditSensor` behaviour (including `audit_exported`, while/outfeed rules, `get_operations`, checkify edges), not only import smoke. |

## Near term (next few releases)

Focus: **integrations, ergonomics, and trust** without bloating `jaxlint.core`.

1. **Output formats** — **Shipped:** GitHub annotation JSON, **SARIF** 2.1.0 (`--format sarif`), **`compact`** text, and **`rich`** (`--format rich`, table layout; explicit `rich` dependency). **Remaining:** optional further Rich polish (panels, links) if desired.
2. **Hosted docs hygiene** — **`sphinx.fail_on_warning` is already on** in `.readthedocs.yml`. **Remaining:** shrink **`linkcheck_ignore`** in `docs/conf.py` once the canonical RTD/HTML URLs and GitHub targets respond reliably to automated linkcheck (the root Read the Docs URL and GitHub may 404 or rate-limit until the project and links are stable).
3. **CI parity** — **Shipped:** required **`jax-hlo`** job installs `jaxlint[hlo]` and runs `pytest -m needs_jax` (merge-blocking). PR **`lint-test`** jobs run smoke + Ruff; full suite + coverage on **`main`** only.
4. **API reference** — **Shipped:** Sphinx **autodoc** / **autosummary** for `jaxlint.core` (and configuration types under `jaxlint.core.config`; see **`api`** in the docs toctree). **`jaxlint.hlo`** stays a narrative-only page (`hlo`) so JAX is not imported on RTD builders.
5. **HLO credibility** — **Shipped:** targeted pytest for `SemanticAnalyzer` / `GraphAuditSensor` (`get_operations`, `audit_kernel` / `audit_exported`, while and outfeed violation paths, `instrument_and_run` checkify failure). **Shipped (goldens):** StableHLO text fixtures under `tests/fixtures/hlo/` for two **`audit_exported`** checks — **outfeed** side-effect and **`stablehlo.while`** detection — with duck-typed export tests (**`@pytest.mark.needs_jax`**, **`jaxlint[hlo]`** in CI; merge-blocking **`jax-hlo`** pins JAX/jaxlib — see [Changelog](changelog.md)). **Remaining:** broaden golden coverage for more export-heavy cases (and optional broader MLIR parsing) as needed.

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

Open issues or RFC-style PRs that spell out **rule IDs**, **default severity**, **config keys**, and **test fixtures**. Breaking changes should bump minor/major semver and mention migration in the Sphinx **[Changelog](changelog.md)**.

## Current sprint notes (June 2026)

**Theme:** **Ergonomics** — human-readable CLI output without changing machine-oriented formats.

| Deliverable | Status |
|-------------|--------|
| **`--format rich`** for `check` / `doc` (`format_rich`, tests, CLI docs) | **Landed** |
| Trim `linkcheck_ignore` after manual URL verification | **Partial** — `curl -sI` (May 2026): `https://jaxlint.readthedocs.io/` and `https://github.com/jax-ml/jaxlint` both return **404**; ignores kept in `docs/conf.py` with a short comment until those endpoints serve 200. |
| Pin JAX in `jax-hlo` CI to stabilize golden text | **Done** — pins **`jax==0.10.0`** / **`jaxlib==0.10.0`** after `uv sync --all-groups --extra hlo`. |
| Golden StableHLO: `stablehlo.while` for `audit_exported` (`while_loop_detected.mlir` + fake export test) | **Done** |

**Previous sprint (May 2026):** Golden StableHLO fixtures + `needs_jax` wiring, changelog + docs index, roadmap / packaging version narrative (see shipped baseline and [Changelog](changelog.md)).
