# Changelog

Notable changes to jaxlint are listed here. Versions match the `version` field in [`pyproject.toml`](https://github.com/jax-ml/jaxlint/blob/main/pyproject.toml) at release time.

## 0.4.0

- **CLI:** **`--format rich`** now wraps diagnostics in a **panel** with a **severity summary** line (`error=` / `warning=` / `info=` counts), configurable **width** via **`COLUMNS`**, terminal size when interactive, or **120** columns when stdout is not a TTY. **Location** may use **OSC 8** `file:` hyperlinks when stdout is a TTY and **`NO_COLOR`** is unset (plain text when piped or when colors are off). See [CLI](cli.md).

## 0.3.2

- **LSP:** Server-side debounce and cancellation for **`textDocument/didChange`**: environment variable **`JAXLINT_LSP_DEBOUNCE_MS`** (default **200** ms; **`0`** disables debounce and restores the previous immediate, sequential behaviour). Per-file **canonical URI** state: cancel prior debounced **asyncio** work, **generation** guard so superseded runs do not publish, **`didClose`** invalidation and **empty** diagnostics, live buffer read in the **thread worker** after debounce. See [Language server](lsp.md).

## Unreleased

- **LSP (optional):** **`jaxlint[lsp]`** adds a **diagnostics-only** language server (**stdio**, **full** document sync, **`file://`** URIs only). Entrypoints: **`jaxlint-lsp`**, **`python -m jaxlint.lsp`**. Uses **`load_config(path.parent)`** and **`diagnostics_for_python_source`** (no disk cache in the server path). See [Language server](lsp.md).
- **Library:** **`diagnostics_for_python_source`** on **`jaxlint.core`** for path + source string (sorted diagnostics; cache-free).
- **CI:** PR **`lint-test`** syncs with **`--extra lsp`** / **`--extra mcp`** and runs **`pytest -m lsp`** and **`pytest -m mcp`**.
- **Docs / roadmap:** LSP sprint notes (**`jaxlint-sprint-2026-11`**) table aligned with plan IDs (**L01–L06**); formal outcome review **APPROVE-with-nits** (2026-05-05, no blocking DoD gaps). **Open TODOs** for non-blocking LSP v1 nits were closed in **0.2.2** (see [Roadmap](roadmap.md)).
- **Performance / CLI:** Opt-in **disk cache** for **`jaxlint check`**: **`--cache-dir PATH`** and optional **`[tool.jaxlint] cache-dir`** (CLI wins when both are set). Cached entries are keyed by file identity (`mtime`/`size`), lint configuration fingerprint, and **`jaxlint` version**. Writes use atomic replace; POSIX uses **`flock`** on lock files next to cache blobs.
- **CLI:** Invalid **`--jobs` / `-j`** values (for example **`-j 0`**) on **`jaxlint check`** now exit with code **2** and a clear message instead of an uncaught traceback.
- **Docs:** Refreshed **`linkcheck_ignore`** rationale in `docs/conf.py` with **`curl -sI -L`** status evidence (2026-05-05); next scheduled review **2026-10-01** while RTD/GitHub URLs remain **404**.
- **Testing:** Golden StableHLO under [`tests/fixtures/hlo/`](https://github.com/jax-ml/jaxlint/tree/main/tests/fixtures/hlo): outfeed / **`stablehlo.while`** **`audit_exported`** paths, plus **`add_constant_kernel`** / **`multiply_reshape_kernel`** inventory goldens (duck-typed fake exports; **`needs_jax`**). Merge-blocking **jax-hlo** CI installs **`jaxlint[hlo]`** and JAX.
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
