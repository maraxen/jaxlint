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

## Behaviour (v1)

| Topic | Detail |
|-------|--------|
| **Diagnostics** | In-memory lint of the open buffer; results are published as LSP diagnostics. |
| **Document sync** | **Full** text sync (`TextDocumentSyncKind.Full`). |
| **`didClose`** | Publishes **empty diagnostics** for the closed URI so clients clear editor squiggles (same as non-Python / unsupported URIs). |
| **URIs** | **`file://` only**; other schemes are ignored (empty diagnostics). |
| **Configuration** | Resolved with **`load_config(path.parent)`** from the file path — same discovery as the CLI (**`jaxlint.toml`** then **`pyproject.toml`** in each ancestor; native **`[jaxlint]`** or legacy **`[tool.jaxlint]`** in **`jaxlint.toml`**, **`[tool.jaxlint]`** only in **`pyproject.toml`**). |
| **Disk cache** | **Not used.** The LSP path calls **`diagnostics_for_python_source`** with explicit source text and a **`LintConfig`**; there is no cache read/write in the server. |
| **CLI overrides** | There is **no** `--select` / `--ignore` equivalent in the server; rule selection comes from config only. |
| **Debounce** | Per-change republish is immediate (**debouncing is left to the client** or future work). |

## Library hook

For tests or custom tooling, use **`jaxlint.core.diagnostics_for_python_source`**: same diagnostics as **`run_checks`** for a given path and source string, sorted and **without** touching the on-disk cache. See [Library usage](library.md).

## See also

- [CLI reference](cli.md) — terminal workflow and **`jaxlint check`** flags (including cache and overrides the LSP does not expose).
