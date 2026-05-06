# Language Server (`jaxlint[lsp]`)

jaxlint ships an optional **diagnostics-only** language server for editors that speak LSP over **stdio**. This is **v1** scope: publish diagnostics from the same rules as **`jaxlint check`** (performance sensors, doc sections, MathJax). **Formatting, code actions, completions, and HLO analysis are out of scope** for the server.

## Install

The server depends on **[Pygls](https://pygls.readthedocs.io/)** (pulled in by the **`lsp`** extra):

```bash
uv sync --extra lsp
# or when installing the package
pip install 'jaxlint[lsp]'
```

## Run

- **`jaxlint-lsp`** — console script from `[project.scripts]` (same as `python -m jaxlint.lsp`).
- **`python -m jaxlint.lsp`** — module entrypoint.

Wire your editor to launch one of the above with **stdio** transport. There is no TCP or socket mode in this release.

## Environment & lifecycle

### Environment variable

| Variable | Meaning | Default |
|----------|---------|---------|
| **`JAXLINT_LSP_DEBOUNCE_MS`** | Milliseconds to debounce **`textDocument/didChange`** before running diagnostics on the buffer | **200**. Set to **`0`** to disable debounce and use the same sequential, immediate republish path as pre-debounce releases. |

*No other **`JAXLINT_*`** environment variables are read by the server (as of **0.1.0a1**).*

### Lifecycle (not configurable via env)

| Topic | Behaviour |
|-------|-----------|
| **`textDocument/didOpen`** | Diagnostics run **immediately** (no debounce). |
| **`textDocument/didChange`** | Debounced by **`JAXLINT_LSP_DEBOUNCE_MS`**; superseded runs are cancelled via a generation guard. |
| **`textDocument/didClose`** | Publishes **empty diagnostics** for the URI and cancels any pending debounced work so editor squiggles clear. |
| **URIs** | **`file://` only**; other schemes yield empty diagnostics (same as unsupported files). |

## Diagnostics, sync, and configuration

| Topic | Detail |
|-------|--------|
| **Diagnostics** | In-memory lint of the open buffer; results are published as LSP diagnostics. |
| **Document sync** | **Full** text sync (`TextDocumentSyncKind.Full`). |
| **Configuration** | Resolved with **`load_config(path.parent)`** from the file path — same discovery as the CLI (**`jaxlint.toml`** then **`pyproject.toml`** in each ancestor; native **`[jaxlint]`** or legacy **`[tool.jaxlint]`** in **`jaxlint.toml`**, **`[tool.jaxlint]`** only in **`pyproject.toml`**). |
| **Disk cache** | **Not used.** The LSP path calls **`diagnostics_for_python_source`** with explicit source text and a **`LintConfig`**; there is no cache read/write in the server. |
| **CLI overrides** | There is **no** `--select` / `--ignore` equivalent in the server; rule selection comes from config only. |

## Library hook

For tests or custom tooling, use **`jaxlint.core.diagnostics_for_python_source`**: same diagnostics as **`run_checks`** for a given path and source string, sorted and **without** touching the on-disk cache. See [Library usage](library.md).

## See also

- [CLI reference](cli.md) — terminal workflow and **`jaxlint check`** flags (including cache and overrides the LSP does not expose).
