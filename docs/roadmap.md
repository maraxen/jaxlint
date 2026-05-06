# Roadmap

This page is the **canonical** roadmap. The README carries a short pointer so it does not drift from here.

## Product vision

**jaxlint** stays **easy to drive from CI and editors**: a JAX-aware static checker and doc-quality tool whose **library surface** (`jaxlint.core`) stays **import-light** (no JAX) while **`jaxlint[hlo]`** can go deeper where JAX and exports are already available.

## Shipped baseline (aligned with 0.2.x)

This section tracks what is **in the repository today**. **Packaged releases** use the `version` in `pyproject.toml` (currently **0.3.1**); treat that as the source of truth for semver, not an informal “v0.1” label.

Already in tree today:

| Area | Status |
|------|--------|
| **CLI** (`check`, `doc`, `score`, `rules`, `init`, `version`, `hlo`) | Stable for core flows; `check`/`doc` support `--format text`, `compact`, `json`, `github` / `github-annotations`, **SARIF** 2.1.0 (`sarif` / `sarif-2.1.0`), or **`rich`** (table layout). **`check`** supports **`-j` / `--jobs`** for parallel file linting (see [CLI](cli.md)). |
| **Rules** | JL/JD/JM catalogue with fixtures and docs. |
| **Config** | **`[jaxlint]`** in **`jaxlint.toml`** (native) or legacy **`[tool.jaxlint]`** there; **`[tool.jaxlint]`** only in **`pyproject.toml`**. Selects/ignores/per-file ignores and **`severity-overrides`**. |
| **Testing** | Pytest tiers (`smoke` / `unit` / `integration` / `needs_jax` / `lsp` / `mcp`), coverage floor, GitHub Actions. Pull requests run **smoke** + **LSP** (`pytest -m lsp`) + **MCP** (`pytest -m mcp`) + lint; **full pytest + coverage** runs on **`main` pushes only** (`lint-test`). |
| **Docs** | Sphinx + MyST + Read the Docs (HTML plus **PDF** plus **htmlzip** on version builds). |
| **`jaxlint[hlo]`** | Optional imports; CLI only checks the extra — **merge-blocking `needs_jax` tests** cover export → `SemanticAnalyzer` / `GraphAuditSensor` behaviour (including `audit_exported`, while/outfeed rules, `get_operations`, checkify edges), not only import smoke. |

## Near term (next few releases)

Focus: **integrations, ergonomics, and trust** without bloating `jaxlint.core`.

1. **Output formats** — **Shipped:** GitHub annotation JSON, **SARIF** 2.1.0 (`--format sarif`), **`compact`** text, and **`rich`** (`--format rich`, table layout; explicit `rich` dependency). **Remaining:** optional further Rich polish (panels, links) if desired.
2. **Hosted docs hygiene** — **`sphinx.fail_on_warning` is already on** in `.readthedocs.yml`. **Partial (July 2026):** **`curl -sI -L`** (2026-05-05) still returns **404** for `https://jaxlint.readthedocs.io/` and `https://github.com/jax-ml/jaxlint`; **`linkcheck_ignore`** stays until stable **200** (next review **2026-10-01**; see `docs/conf.py` comments).
3. **CI parity** — **Shipped:** required **`jax-hlo`** job installs `jaxlint[hlo]` and runs `pytest -m needs_jax` (merge-blocking). PR **`lint-test`** jobs run smoke + Ruff + **`pytest -m lsp`** + **`pytest -m mcp`** (with **`uv sync --extra lsp --extra mcp`**); full suite + coverage on **`main`** only.
4. **API reference** — **Shipped:** Sphinx **autodoc** / **autosummary** for `jaxlint.core` (and configuration types under `jaxlint.core.config`; see **`api`** in the docs toctree). **`jaxlint.hlo`** stays a narrative-only page (`hlo`) so JAX is not imported on RTD builders.
5. **HLO credibility** — **Shipped:** targeted pytest for `SemanticAnalyzer` / `GraphAuditSensor` (`get_operations`, `audit_kernel` / `audit_exported`, while and outfeed violation paths, `instrument_and_run` checkify failure). **Shipped (goldens):** StableHLO text fixtures under `tests/fixtures/hlo/` — **outfeed** / **`stablehlo.while`** **`audit_exported`** paths plus **add+constant** and **multiply+reshape** inventory goldens (duck-typed fake exports; **`@pytest.mark.needs_jax`**, **`jaxlint[hlo]`** in CI; merge-blocking **`jax-hlo`** — see [Changelog](changelog.md)). **Remaining:** broaden golden coverage for more export-heavy cases (and optional broader MLIR parsing) as needed.

## Medium term

| Initiative | Goal |
|------------|------|
| **Language Server (LSP)** | **Shipped (v1):** optional **`jaxlint[lsp]`** — Pygls **stdio** server, **full** sync, **`file://`** only, **`load_config(path.parent)`**, **`diagnostics_for_python_source`** (no disk cache). **Remaining:** debounce/client policy, richer LSP features only if they stay thin (formatting still out of scope for this server). |
| **HLO diagnostics** | Optional analysers over lowered IR/text (similar in spirit to “HLO dump regex” tooling described in jaxbeans discussions): scatter→gather fusion gaps, noisy kernel launches — scoped to **`jaxlint.hlo`**. |
| **Performance at scale** | **Shipped (May 2026):** opt-in disk cache via **`jaxlint check --cache-dir`** and **`[tool.jaxlint] cache-dir`** (serialized diagnostics; config fingerprint + file stats). **Shipped (Aug 2026):** single parse per file on `check`, `run_checks(..., jobs=)` parallelism. |
| **Config evolution** | **Shipped (May 2026):** **`severity-overrides`** under **`[tool.jaxlint]`** / **`[jaxlint]`**, optional **`jaxlint.toml`** discovery + **`--config` file**, **`CACHE_FORMAT_VERSION`** bump with fingerprinted overrides; SARIF/GitHub levels follow post-override **`Diagnostic.severity`**. **Shipped (0.3.1):** native **`[jaxlint]`** table in **`jaxlint.toml`** with legacy **`[tool.jaxlint]`** compatibility (mutually exclusive in-file). **Remaining:** richer SARIF rule metadata / org severity policies if needed. |

## Longer horizon / exploratory

These are **not committed** milestones; revisit when upstream JAX and IDE ecosystems move.

| Idea | Notes |
|------|--------|
| **MCP server** | **v1 (0.2.3):** optional **`jaxlint[mcp]`** — FastMCP stdio server (`jaxlint-mcp`), tools **`jaxlint_check`** (structured envelope, **`jobs`** / **`cache_dir`** on paths), **`jaxlint_capabilities`**, **`jaxlint_rules`**, **`jaxlint_version`** (see [Changelog](changelog.md)). |
| **Math semantics** | Beyond delimiter/LaTeX structure: optional SymPy-backed checks — opt-in dependency. |
| **Deeper parallelism / sharding UX** | New rules aligned with JAX `jit`/`shard_map` ergonomics as patterns stabilize. |

## How to steer this roadmap

Open issues or RFC-style PRs that spell out **rule IDs**, **default severity**, **config keys**, and **test fixtures**. Breaking changes should bump minor/major semver and mention migration in the Sphinx **[Changelog](changelog.md)**.

## Current sprint notes (May 2026 — Config evolution)

**Theme:** **Per-rule severity overrides**, **`jaxlint.toml`** / **`--config` file**, cache namespace **v2**, SARIF/GitHub aligned with effective severity.

**task_id:** `jaxlint-sprint-2026-05`

**Artifact:** `.praxia/sprint-2026-05-config-evolution.md`

| ID | Deliverable | Status |
|----|-------------|--------|
| **CF1** | **`severity-overrides`** + **`LintConfig`** parsing / validation | **Shipped** |
| **CF2** | **`effective_severity`** after **`rule_enabled`** (overrides win vs JL005 **`strict-shapes`**) | **Shipped** |
| **CF3** | **`CACHE_FORMAT_VERSION = 2`** + `_canonical_config_dict` includes overrides | **Shipped** |
| **CF4** | SARIF / GitHub JSON use runner **`Diagnostic.severity`** | **Shipped** |
| **CF5** | **`jaxlint.toml`** discovery; **`--config`** file vs directory; same-dir conflict **error** | **Shipped** |
| **CF6** | Docs (`configuration.md`, `cli.md`), changelog **0.3.0**, tests | **Shipped** |

**Outcome review:** **APPROVE** (after remediation: cache fingerprint regression test for severity-only override delta).

## Current sprint notes (May 2026 — Agent-grade MCP v1 / “nascency”)

**Theme:** **Structured MCP responses**, **`run_checks`** passthrough options from agents, and a **`jaxlint_capabilities`** introspection tool — backward compatible by default.

**task_id:** `jaxlint-sprint-2026-05-nascency`

| ID | Deliverable | Status |
|----|-------------|--------|
| **MC1** | **`jaxlint_check`**: optional **`structured`**, **`jobs`**, **`cache_dir`**; legacy default unchanged | **Shipped** |
| **MC2** | **`jaxlint.mcp.v1`** envelope (`success`, `diagnostics`, `error`, `meta` with wall-clock **`elapsed_ms`**) | **Shipped** |
| **MC3** | **`jaxlint_capabilities`** + **`MCP_TOOLS_REGISTRY`** (single source of truth) | **Shipped** |
| **MC4** | **`pytest -m mcp`** — structured path/source, JD001, unsupported options, invalid **`jobs`**, capabilities JSON | **Shipped** |
| **MC5** | Docs + version **0.2.3** (changelog, README, roadmap) | **Shipped** |

## Current sprint notes (May 2026 — LSP nits + MCP preview)

**Theme:** **LSP test/CI/doc alignment** (markers, `didClose` coverage, contributing truth) plus an optional **MCP** stdio preview for agent workflows.

**task_id:** `jaxlint-sprint-2026-05-oda`

| ID | Deliverable | Status |
|----|-------------|--------|
| **N1** | Module-level **`lsp`** marker on **`tests/test_lsp_convert.py`** and **`tests/test_diagnostics_for_python_source.py`** | **Shipped** |
| **N2** | CI **`uv run pytest -q -m lsp`** (no hard-coded LSP file list) | **Shipped** |
| **N3** | Stdio test: **`textDocument/didClose`** → empty **`publishDiagnostics`** | **Shipped** |
| **N4** | **`docs/lsp.md`**: **`didClose`** clears diagnostics | **Shipped** |
| **N5** | **`docs/contributing.md`**: LSP workflow = **`pytest -m lsp`** | **Shipped** |
| **M1** | **`jaxlint[mcp]`** + **`jaxlint-mcp`** / **`python -m jaxlint.mcp`** (FastMCP tools) | **Shipped** |
| **M2** | **`pytest -m mcp`**, CI sync both extras, changelog + README | **Shipped** |

## Current sprint notes (November 2026 — LSP v1)

**Theme:** **Diagnostics-only language server** — optional **`jaxlint[lsp]`** + editor stdio integration without JAX/HLO scope creep.

**task_id:** `jaxlint-sprint-2026-11`

| ID | Deliverable | Status |
|----|-------------|--------|
| **L01** | Optional **`[lsp]`** extra (Pygls), **`jaxlint-lsp`** / **`python -m jaxlint.lsp`** | **Shipped** |
| **L02** | **`parse_python_source`** + **`diagnostics_for_python_source`** (sorted; same per-file semantics as **`_diagnostics_for_file`**); LSP never uses disk cache | **Shipped** |
| **L03** | **`jaxlint.lsp.convert`**: **`Diagnostic`** → LSP types; unit tests | **Shipped** |
| **L04** | **`jaxlint.lsp`**: stdio server, **Full** sync, **`file://`** only, **`didOpen`/`didChange`/`didClose`**, **`asyncio.to_thread`**; **`load_config(path.parent)`** (no CLI **`--select`** overrides) | **Shipped** |
| **L05** | Subprocess JSON-RPC tests (**`publishDiagnostics`** on open/change/**close**); **`lsp`** + **`integration`** markers | **Shipped** |
| **L06** | CI (**`lint-test`** syncs **`lsp`** + **`mcp`** extras; **`pytest -m lsp`** and **`pytest -m mcp`** on PRs), **`docs/lsp.md`**, changelog, library/README pointers | **Shipped** |
| **Review** | Outcome audit (**2026-05-05**): **APPROVE-with-nits** — no blocking gaps vs DoD; follow-ups tracked below | **Recorded** |

### Open TODOs (LSP v1 reviewer nits)

Non-blocking polish from the outcome review; not duplicated as GitHub issues in-repo.

- [x] **LSP tests:** add subprocess/integration coverage that **`textDocument/didClose`** triggers **`publishDiagnostics`** with an **empty** list (today only open/change are asserted).
- [x] **Markers:** either add **`@pytest.mark.lsp`** to **`tests/test_lsp_convert.py`** so **`pytest -m lsp`** runs convert + stdio suites, **or** document in **`docs/contributing.md`** that CI names those files explicitly and **`lsp`** is stdio-focused.
- [x] **Contributing docs:** add a short **Language server (`jaxlint[lsp]`)** subsection under [Contributing](contributing.md) (`uv sync --extra lsp`, suggested pytest invocations, marker meaning).

## Current sprint notes (May 2026 — Disk cache / P09)

**Theme:** **Opt-in disk cache** for `jaxlint check` — roadmap **P09** from the August scale sprint (now delivered).

**task_id:** `jaxlint-sprint-2026-05-p09-cache` (artifact: `.praxia/sprint-2026-05-disk-cache.md`)

| ID | Deliverable | Status |
|----|-------------|--------|
| **P09a** | `--cache-dir` on `check`; `run_checks(..., cache_dir=)` | **Shipped** |
| **P09b** | `jaxlint.core.cache` — fingerprint, JSON envelope, atomic store | **Shipped** |
| **P09c** | Runner wraps per-file diagnostics with cache get/put | **Shipped** |
| **P09d** | `[tool.jaxlint] cache-dir`; CLI overrides TOML | **Shipped** |
| **P09e** | POSIX `flock` on writes; fail-open on corrupt/missing entries | **Shipped** |
| **P09f** | Tests: warm/cold, parallel, invalidation, namespace, CLI, corrupt blob | **Shipped** |

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
| **P09** | Opt-in disk cache (`--cache-dir`) | **Shipped** (May 2026 sprint; see above) |

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
