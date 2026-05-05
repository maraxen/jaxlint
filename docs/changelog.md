# Changelog

Notable changes to jaxlint are listed here. Versions match the `version` field in [`pyproject.toml`](https://github.com/jax-ml/jaxlint/blob/main/pyproject.toml) at release time.

## Unreleased

- **LSP (optional):** **`jaxlint[lsp]`** adds a **diagnostics-only** language server (**stdio**, **full** document sync, **`file://`** URIs only). Entrypoints: **`jaxlint-lsp`**, **`python -m jaxlint.lsp`**. Uses **`load_config(path.parent)`** and **`diagnostics_for_python_source`** (no disk cache in the server path). See [Language server](lsp.md).
- **Library:** **`diagnostics_for_python_source`** on **`jaxlint.core`** for path + source string (sorted diagnostics; cache-free).
- **CI:** PR **`lint-test`** syncs with **`--extra lsp`** and runs LSP-focused tests.
- **Docs / roadmap:** LSP sprint notes (**`jaxlint-sprint-2026-11`**) table aligned with plan IDs (**L01–L06**); formal outcome review **APPROVE-with-nits** (2026-05-05, no blocking DoD gaps). **Open TODOs** for non-blocking LSP v1 nits: checklist in [Roadmap](roadmap.md) (November 2026 sprint).
- **Performance / CLI:** Opt-in **disk cache** for **`jaxlint check`**: **`--cache-dir PATH`** and optional **`[tool.jaxlint] cache-dir`** (CLI wins when both are set). Cached entries are keyed by file identity (`mtime`/`size`), lint configuration fingerprint, and **`jaxlint` version**. Writes use atomic replace; POSIX uses **`flock`** on lock files next to cache blobs.
- **CLI:** Invalid **`--jobs` / `-j`** values (for example **`-j 0`**) on **`jaxlint check`** now exit with code **2** and a clear message instead of an uncaught traceback.
- **Docs:** Refreshed **`linkcheck_ignore`** rationale in `docs/conf.py` with **`curl -sI -L`** status evidence (2026-05-05); next scheduled review **2026-10-01** while RTD/GitHub URLs remain **404**.
- **Testing:** Golden StableHLO under [`tests/fixtures/hlo/`](https://github.com/jax-ml/jaxlint/tree/main/tests/fixtures/hlo): outfeed / **`stablehlo.while`** **`audit_exported`** paths, plus **`add_constant_kernel`** / **`multiply_reshape_kernel`** inventory goldens (duck-typed fake exports; **`needs_jax`**). Merge-blocking **jax-hlo** CI installs **`jaxlint[hlo]`** and JAX.
- **CI:** The **`jax-hlo`** workflow pins **`jax==0.10.0`** and **`jaxlib==0.10.0`** after `uv sync` so **`needs_jax`** runs stay reproducible.
- **CLI:** `jaxlint check` and `jaxlint doc` accept **`--format rich`** (Rich table: severity, rule, location, message). Depends on explicit **`rich`** (see `pyproject.toml`).
- **Docs:** Added this changelog (per [roadmap steering](roadmap.md)).

## 0.2.1

- **Performance:** Single `read_text` + single `ast.parse` per file when both performance sensors and doc/MathJax checks run (`run_checks` combined path).
- **Library / CLI:** `run_checks(..., jobs=...)` with default `min(32, os.process_cpu_count() or 1)`; **`jobs=1`** serial; **`jobs>1`** uses a thread pool over files. **`jaxlint check`** exposes **`-j` / `--jobs`**. Return value is **`list[Diagnostic]` sorted by `Diagnostic.sort_key()`** (stable ordering contract).
- **Internals:** `ParsedPythonFile` / `load_parsed_python_file` in `jaxlint.core.parse_context`; perf sensors take `(tree, path)` (no internal parse).

## 0.2.0

Packaged release line aligned with the shipped CLI, rules engine, SARIF / compact / GitHub diagnostic formats, Sphinx docs (including API reference for `jaxlint.core`), and optional **`jaxlint[hlo]`** analysis with merge-blocking `needs_jax` coverage. See the **Shipped baseline** section in the [Roadmap](roadmap.md).
