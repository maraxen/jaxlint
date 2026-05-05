# Roadmap

This page is the **canonical** roadmap. The README carries a short pointer so it does not drift from here.

## Product vision

**jaxlint** stays **easy to drive from CI and editors**: a JAX-aware static checker and doc-quality tool whose **library surface** (`jaxlint.core`) stays **import-light** (no JAX) while **`jaxlint[hlo]`** can go deeper where JAX and exports are already available.

## Shipped baseline (aligned with 0.2.x)

This section tracks what is **in the repository today**. **Packaged releases** use the `version` in `pyproject.toml` (currently **0.2.1**); treat that as the source of truth for semver, not an informal “v0.1” label.

Already in tree today:

| Area | Status |
|------|--------|
| **CLI** (`check`, `doc`, `score`, `rules`, `init`, `version`, `hlo`) | Stable for core flows; `check`/`doc` support `--format text`, `compact`, `json`, `github` / `github-annotations`, **SARIF** 2.1.0 (`sarif` / `sarif-2.1.0`), or **`rich`** (table layout). **`check`** supports **`-j` / `--jobs`** for parallel file linting (see [CLI](cli.md)). |
| **Rules** | JL/JD/JM catalogue with fixtures and docs. |
| **Config** | `[tool.jaxlint]` parsing, selects/ignores/per-file ignores. |
| **Testing** | Pytest tiers (`smoke` / `unit` / `integration` / `needs_jax`), coverage floor, GitHub Actions. Pull requests run **smoke** + lint; **full pytest + coverage** runs on **`main` pushes only** (`lint-test`). |
| **Docs** | Sphinx + MyST + Read the Docs (HTML plus **PDF** plus **htmlzip** on version builds). |
| **`jaxlint[hlo]`** | Optional imports; CLI only checks the extra — **merge-blocking `needs_jax` tests** cover export → `SemanticAnalyzer` / `GraphAuditSensor` behaviour (including `audit_exported`, while/outfeed rules, `get_operations`, checkify edges), not only import smoke. |

## Near term (next few releases)

Focus: **integrations, ergonomics, and trust** without bloating `jaxlint.core`.

1. **Output formats** — **Shipped:** GitHub annotation JSON, **SARIF** 2.1.0 (`--format sarif`), **`compact`** text, and **`rich`** (`--format rich`, table layout; explicit `rich` dependency). **Remaining:** optional further Rich polish (panels, links) if desired.
2. **Hosted docs hygiene** — **`sphinx.fail_on_warning` is already on** in `.readthedocs.yml`. **Partial (July 2026):** **`curl -sI -L`** (2026-05-05) still returns **404** for `https://jaxlint.readthedocs.io/` and `https://github.com/jax-ml/jaxlint`; **`linkcheck_ignore`** stays until stable **200** (next review **2026-10-01**; see `docs/conf.py` comments).
3. **CI parity** — **Shipped:** required **`jax-hlo`** job installs `jaxlint[hlo]` and runs `pytest -m needs_jax` (merge-blocking). PR **`lint-test`** jobs run smoke + Ruff; full suite + coverage on **`main`** only.
4. **API reference** — **Shipped:** Sphinx **autodoc** / **autosummary** for `jaxlint.core` (and configuration types under `jaxlint.core.config`; see **`api`** in the docs toctree). **`jaxlint.hlo`** stays a narrative-only page (`hlo`) so JAX is not imported on RTD builders.
5. **HLO credibility** — **Shipped:** targeted pytest for `SemanticAnalyzer` / `GraphAuditSensor` (`get_operations`, `audit_kernel` / `audit_exported`, while and outfeed violation paths, `instrument_and_run` checkify failure). **Shipped (goldens):** StableHLO text fixtures under `tests/fixtures/hlo/` — **outfeed** / **`stablehlo.while`** **`audit_exported`** paths plus **add+constant** and **multiply+reshape** inventory goldens (duck-typed fake exports; **`@pytest.mark.needs_jax`**, **`jaxlint[hlo]`** in CI; merge-blocking **`jax-hlo`** — see [Changelog](changelog.md)). **Remaining:** broaden golden coverage for more export-heavy cases (and optional broader MLIR parsing) as needed.

## Medium term

| Initiative | Goal |
|------------|------|
| **Language Server (LSP)** | Thin wrapper that streams **`Diagnostic`** objects to editors (diagnostics-only v1; formatting out of scope). |
| **HLO diagnostics** | Optional analysers over lowered IR/text (similar in spirit to “HLO dump regex” tooling described in jaxbeans discussions): scatter→gather fusion gaps, noisy kernel launches — scoped to **`jaxlint.hlo`**. |
| **Performance at scale** | **In progress (Aug 2026):** single parse per file on `check`, `run_checks(..., jobs=)` parallelism; **deferred:** opt-in disk cache (`--cache-dir`, next sprint). |
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

## Current sprint notes (August 2026 — Scale & performance)

**Theme:** **Scale & performance** — cut redundant I/O/AST work on `jaxlint check`, file-level parallelism, documented `run_checks` sort contract.

**task_id:** `jaxlint-sprint-2026-08-scale` (execution artifact: `.agents/REFACTOR_ROADMAP`)

| ID | Deliverable | Status |
|----|-------------|--------|
| **P01** | `ParsedPythonFile` / `load_parsed_python_file` (`jaxlint.core.parse_context`) | **Shipped** |
| **P02** | Perf sensors `(tree, path)`; one parse in `perf_diagnostics_for_file` | **Shipped** |
| **P03** | `doc_diagnostics_from_source` + thin `doc_diagnostics_for_file` | **Shipped** |
| **P04** | `run_checks` combined read/parse; sorted return; `jobs` + `ThreadPoolExecutor` | **Shipped** |
| **P05** | `tests/test_parse_budget.py` | **Shipped** |
| **P06** | CLI `-j` / `--jobs` on `check` | **Shipped** |
| **P07** | `tests/test_runner_parallel.py` | **Shipped** |
| **P08** | Docs: cli, changelog, roadmap, library | **Shipped** |
| **P09** | Opt-in disk cache (`--cache-dir`) | **Deferred** (next sprint) |

## Current sprint notes (July 2026 — Trust surfaces)

**Theme:** **Trust surfaces** — docs link hygiene evidence, HLO golden coverage (strategy B), roadmap truthfulness.

| ID | Deliverable | Status |
|----|-------------|--------|
| **D02** | URL verification (`curl -sI -L` for RTD + GitHub + `linkcheck_ignore` targets) | **Done** — both canonical URLs return **HTTP/2 404** (2026-05-05); documented in `docs/conf.py`. |
| **D01** | `linkcheck_ignore` policy (DoD branch **ii** unless stable **200** for both) | **Partial** — ignores retained; comments refreshed with evidence; **next review 2026-10-01**. |
| **H01** | Golden MLIR: add + constants (`SemanticAnalyzer` inventory path) | **Done** — `tests/fixtures/hlo/add_constant_kernel.mlir`. |
| **H02** | Golden MLIR: multiply + reshape + constants | **Done** — `tests/fixtures/hlo/multiply_reshape_kernel.mlir`. |
| **H03** | `needs_jax` + `unit` tests on fake-export duck type (`get_operation_inventory` / `get_operations`) | **Done** — `tests/test_hlo_semantics.py`. |
| **R01** | README blurb alignment | **R01 deferred** (no change this sprint). |

## Previous sprint (June 2026 — Ergonomics)

**Theme:** **Ergonomics** — human-readable CLI output without changing machine-oriented formats. **Outcome:** **Completed** for Rich CLI; **partial** for hosted-docs link trim (same **404** evidence as July).

| Deliverable | Status |
|-------------|--------|
| **`--format rich`** for `check` / `doc` (`format_rich`, tests, CLI docs) | **Landed** |
| Trim `linkcheck_ignore` after manual URL verification | **Partial** — superseded by July 2026 D01/D02 evidence in `docs/conf.py`. |
| Pin JAX in `jax-hlo` CI to stabilize golden text | **Done** — pins **`jax==0.10.0`** / **`jaxlib==0.10.0`** after `uv sync --all-groups --extra hlo`. |
| Golden StableHLO: `stablehlo.while` for `audit_exported` (`while_loop_detected.mlir` + fake export test) | **Done** |

**Previous sprint (May 2026):** Golden StableHLO fixtures + `needs_jax` wiring, changelog + docs index, roadmap / packaging version narrative (see shipped baseline and [Changelog](changelog.md)).
