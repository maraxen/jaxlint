# Changelog

Notable changes to jaxlint are listed here. Versions match the `version` field in [`pyproject.toml`](https://github.com/maraxen/jaxlint/blob/main/pyproject.toml) at release time.

## 0.1.0a1

- **Pre-release (alpha):** **PEP 440** version **`0.1.0a1`** — first PyPI-oriented line still under **`Development Status :: 3 - Alpha`**. Use **`pip install --pre jaxlint`** or pin **`jaxlint==0.1.0a1`** until a stable **`0.1.0`**.
- **License:** jaxlint is distributed under the **MIT License** (see repository `LICENSE`).
- **Packaging / metadata:** PyPI runway — **`[project.urls]`**, Python **3.11** / **3.12** classifiers, **`Typing :: Typed`**, reconciled authors — semver reset to the **0.1.x** pre-release line.
- **Historical numbering:** Sections labeled **0.3.x**–**0.4.x** below document development history prior to the **0.1.x** public-metadata reset; treat them as archival changelog sections, not distinct PyPI lines unless that version was already published.
- **Library / CLI:** Default **`-j`** / **`run_checks(..., jobs=None)`** uses **`os.process_cpu_count()`** on Python **3.13+** and falls back to **`os.cpu_count()`** on **3.11** / **3.12** (avoids **`AttributeError`** in CI and on older runtimes).
- **Docs:** Sphinx **linkcheck** ignores the **OASIS SARIF** spec URL (often returns **403** to automated clients).

## 0.4.1

- **SARIF:** Each run includes **`tool.driver.rules`**: one **reportingDescriptor** per **distinct** `ruleId` in **`results`**, ordered by first occurrence in **`Diagnostic.sort_key()`** order. Known ids use the built-in catalogue (**`shortDescription`**, **`defaultConfiguration.level`** = default rule severity). Unknown ids use **`shortDescription`** `"Unknown or custom rule"` and omit **`defaultConfiguration`**. Every **`result`** sets **`ruleIndex`** into that **`rules`** array. **`results[].level`** stays the **effective** severity (after **`severity-overrides`** and related logic); descriptor defaults document the catalogue only.
- **Config:** Optional **`sarif-rule-docs-base`** (**HTTPS** URL string) under **`[jaxlint]`** or **`[tool.jaxlint]`**. When set, each rule descriptor gets **`helpUri`** = base (trailing slashes trimmed) + **`/`** + **`rule_id`**. Empty or non-HTTPS values are rejected at parse time. Programmatic SARIF must pass **`cfg`** into **`format_sarif`** / **`format_diagnostics`** for this (see [Library](library.md)).
- **Testing:** Tests marked **`needs_jax`** use **`pytest-timeout`** (**60s** per **test function** only — **`timeout_func_only`** in `pyproject.toml`; markers applied in **`tests/conftest.py`**).

## 0.4.0

- **CLI:** **`--format rich`** now wraps diagnostics in a **panel** with a **severity summary** line (`error=` / `warning=` / `info=` counts), configurable **width** via **`COLUMNS`**, terminal size when interactive, or **120** columns when stdout is not a TTY. **Location** may use **OSC 8** `file:` hyperlinks when stdout is a TTY and **`NO_COLOR`** is unset (plain text when piped or when colors are off). See [CLI](cli.md).

## 0.3.2

- **LSP:** Server-side debounce and cancellation for **`textDocument/didChange`**: environment variable **`JAXLINT_LSP_DEBOUNCE_MS`** (default **200** ms; **`0`** disables debounce and restores the previous immediate, sequential behaviour). Per-file **canonical URI** state: cancel prior debounced **asyncio** work, **generation** guard so superseded runs do not publish, **`didClose`** invalidation and **empty** diagnostics, live buffer read in the **thread worker** after debounce. See [Language server](lsp.md).

## Unreleased

- **Dev / CI:** **[Astral ty](https://docs.astral.sh/ty/)** — dev dependency, **`[tool.ty]`** in **`pyproject.toml`**, and **`lint-test`** runs **`uv run ty check --output-format github`** on first-party **`src/`**; see [Contributing](contributing.md#static-typing-ty). Typing-only fixes: Griffe **`Docstring.parse`** style argument (`style.py`); explicit AST narrowing in **`scatter_gather.py`** (`_at_set_base`).
- **Docs / HLO:** [Goldens: mapping vs counts](hlo.md#goldens-mapping-vs-counts) explains **`get_operation_inventory`** vs **`get_source_mapping`** vs **`get_operations`**; [ADR 0001](adr/0001-hlo-diagnostics-feasibility.md) tracks Phase A feasibility before **`HLO*`** rules.
- **LSP (optional):** **`jaxlint[lsp]`** adds a **diagnostics-only** language server (**stdio**, **full** document sync, **`file://`** URIs only). Entrypoints: **`jaxlint-lsp`**, **`python -m jaxlint.lsp`**. Uses **`load_config(path.parent)`** and **`diagnostics_for_python_source`** (no disk cache in the server path). See [Language server](lsp.md).
- **Library:** **`diagnostics_for_python_source`** on **`jaxlint.core`** for path + source string (sorted diagnostics; cache-free).
- **CI:** PR **`lint-test`** syncs with **`--extra lsp`** / **`--extra mcp`** and runs **full pytest + coverage** (same gate as **main**).
- **Docs / roadmap:** LSP sprint notes (**`jaxlint-sprint-2026-11`**) table aligned with plan IDs (**L01–L06**); formal outcome review **APPROVE-with-nits** (2026-05-05, no blocking DoD gaps). **Open TODOs** for non-blocking LSP v1 nits were closed in **0.2.2** (see [Roadmap](roadmap.md)).
- **Performance / CLI:** Opt-in **disk cache** for **`jaxlint check`**: **`--cache-dir PATH`** and optional **`[tool.jaxlint] cache-dir`** (CLI wins when both are set). Cached entries are keyed by file identity (`mtime`/`size`), lint configuration fingerprint, and **`jaxlint` version**. Writes use atomic replace; POSIX uses **`flock`** on lock files next to cache blobs.
- **CLI:** Invalid **`--jobs` / `-j`** values (for example **`-j 0`**) on **`jaxlint check`** now exit with code **2** and a clear message instead of an uncaught traceback.
- **Docs:** Roadmap **Canonical hosted URLs** table plus refreshed **`linkcheck_ignore`** comments in `docs/conf.py` (**`curl -sI -L`**, **2026-05-06**); canonical RTD/GitHub URLs still **404**; next review **2026-10-01**.
- **Testing:** Golden StableHLO under [`tests/fixtures/hlo/`](https://github.com/maraxen/jaxlint/tree/main/tests/fixtures/hlo): outfeed / **`stablehlo.while`** **`audit_exported`** paths, plus **`add_constant_kernel`** / **`multiply_reshape_kernel`** inventory goldens (duck-typed fake exports; **`needs_jax`**). Merge-blocking **jax-hlo** CI installs **`jaxlint[hlo]`** and JAX.
- **CI:** On **pull requests**, the **`docs`** job runs only when **`docs/**`**, **`.readthedocs.yml`**, or **`README.md`** change; **`main`** / **`master`** pushes always run it (see workflow header comment for frozen globs).
- **CI:** The **`jax-hlo`** workflow pins **`jax==0.10.0`** and **`jaxlib==0.10.0`** after `uv sync` so **`needs_jax`** runs stay reproducible.
- **CLI:** `jaxlint check` and `jaxlint doc` accept **`--format rich`** (Rich table: severity, rule, location, message). Depends on explicit **`rich`** (see `pyproject.toml`).
- **Docs:** Added this changelog (per [roadmap steering](roadmap.md)).

## 0.3.1

- **Configuration:** Native **`[jaxlint]`** table in **`jaxlint.toml`** (recommended). Legacy **`[tool.jaxlint]`** remains valid in **`jaxlint.toml`**; **both sections in the same file** is a **load-time error**. **`pyproject.toml`** uses **`[tool.jaxlint]`** only — a top-level **`[jaxlint]`** table is rejected. **`jaxlint init`** emits **`[jaxlint]`**. Equivalent native vs legacy **`jaxlint.toml`** yields identical **`LintConfig`** (disk cache **`CACHE_FORMAT_VERSION`** unchanged).

## 0.3.0

- **Configuration:** Optional **`jaxlint.toml`** in each directory is preferred over **`pyproject.toml`** during upward discovery; both in the same directory is a **load-time error**. **`--config`** may point at a **`jaxlint.toml`** or **`pyproject.toml`** file (parse in place, no walk) or at a directory (walk). Relative **`cache-dir`** and other paths resolve against the **directory containing the loaded config file**.
- **Severity:** **`severity-overrides`** under **`[tool.jaxlint]`** (or **`[tool.jaxlint.severity-overrides]`**) maps rule patterns to **`error` / `warning` / `info`**. Overrides apply **after** sensors and **`rule_enabled`**, so they win over **JL005** **`strict-shapes`** promotions. Unknown override keys and invalid severities fail at **parse** time. Pattern matching matches **`select` / `ignore`** (`pattern_matches`).
- **Cache:** **`CACHE_FORMAT_VERSION`** bumped to **`2`**; canonical config fingerprint includes **`severity-overrides`** (sorted keys).
- **Exports:** SARIF and GitHub annotations JSON use **`Diagnostic.severity`** as emitted by the runner (post-overrides); no separate severity remap in formatters.
- **Docs / CLI:** Updated [configuration](configuration.md) and [CLI reference](cli.md); **`starter_toml_snippet()`** mentions **`severity-overrides`**.

## 0.2.3

- **MCP (agent-grade v1):** **`jaxlint_check`** accepts optional **`structured`** (JSON envelope **`jaxlint.mcp.v1`** with **`success`**, **`diagnostics`**, **`error`**, **`meta`** including **`elapsed_ms`** and **`paths_resolved`**), **`jobs`**, and **`cache_dir`** (path mode only; forwarded to **`run_checks`**). Default **`structured=false`** preserves legacy behaviour (diagnostic array JSON or **`error:`** strings). Source mode rejects **`jobs`** / **`cache_dir`** with **`unsupported_option`** when structured. Invalid **`jobs`** maps to **`invalid_argument`** (legacy **`error:`** + message).
- **MCP:** New **`jaxlint_capabilities`** tool — JSON document with **`schema_version`**, **`package_version`**, and registered tool metadata (**`MCP_TOOLS_REGISTRY`** single source of truth).
- **Tests / docs:** **`pytest -m mcp`** covers structured envelopes, option validation, and capabilities; README + roadmap sprint notes updated.

## 0.2.2

- **LSP:** Module-level **`pytest.mark.lsp`** on **`tests/test_lsp_convert.py`** and **`tests/test_diagnostics_for_python_source.py`**; PR CI runs **`uv run pytest -q -m lsp`**. Subprocess coverage for **`textDocument/didClose`** publishing **empty** diagnostics; **`docs/lsp.md`** documents the clear behaviour. **Contributing** documents the **`pytest -m lsp`** workflow. Roadmap **Open TODOs (LSP v1 reviewer nits)** checked off where satisfied (see [Roadmap](roadmap.md)).
- **MCP (preview):** Optional **`jaxlint[mcp]`** installs the official **`mcp`** SDK. Console script **`jaxlint-mcp`** and **`python -m jaxlint.mcp`** run a **stdio** FastMCP server exposing **`jaxlint_check`** (paths or in-memory `source` + `virtual_path`), **`jaxlint_rules`**, and **`jaxlint_version`**. Tests: **`pytest -q -m mcp`** (CI **`lint-test`** syncs **`--extra lsp --extra mcp`**).

## 0.2.1

- **Performance:** Single `read_text` + single `ast.parse` per file when both performance sensors and doc/MathJax checks run (`run_checks` combined path).
- **Library / CLI:** `run_checks(..., jobs=...)` with default `min(32, os.process_cpu_count() or 1)`; **`jobs=1`** serial; **`jobs>1`** uses a thread pool over files. **`jaxlint check`** exposes **`-j` / `--jobs`**. Return value is **`list[Diagnostic]` sorted by `Diagnostic.sort_key()`** (stable ordering contract).
- **Internals:** `ParsedPythonFile` / `load_parsed_python_file` in `jaxlint.core.parse_context`; perf sensors take `(tree, path)` (no internal parse).

## 0.2.0

Packaged release line aligned with the shipped CLI, rules engine, SARIF / compact / GitHub diagnostic formats, Sphinx docs (including API reference for `jaxlint.core`), and optional **`jaxlint[hlo]`** analysis with merge-blocking `needs_jax` coverage. See the **Shipped baseline** section in the [Roadmap](roadmap.md).
