# jaxlint

Static analysis and docstring quality tooling for JAX-heavy Python codebases.

## Install

```bash
uv tool install jaxlint
# or from source
uv sync
uv run jaxlint --help
```

Optional HLO / export analysis:

```bash
uv sync --extra hlo
```

## Documentation

Published builds (**HTML**, **PDF**, **htmlzip** per version):

- **[Read the Docs](https://jaxlint.readthedocs.io/)** — import this repository using [`.readthedocs.yml`](.readthedocs.yml); the canonical URL assumes project slug **`jaxlint`**. Pull-request previews are **HTML-only**; PDF/htmlzip artifacts appear on **`latest`/tag/version** builds ([RTD `formats`](https://docs.readthedocs.io/en/stable/config-file/v2.html)).
- **Markdown sources:** [docs/index.md](docs/index.md) (notebook-free Sphinx + MyST stack, aligned with JAX’s docs toolchain at smaller scope).

## CLI

- `jaxlint check [PATHS]...` — performance sensors (AST) plus doc section and MathJax checks by default
- `jaxlint doc [PATHS]...` — doc sections + MathJax only
- `jaxlint score [PATHS]...` — rubric scores for docstrings (optional `--jsonl`, `--baseline`)
- `jaxlint rules` — list rule IDs
- `jaxlint init` — print a starter `[tool.jaxlint]` block
- `jaxlint hlo` — hints for optional `jaxlint.hlo` (requires `--extra hlo`)

**Language server:** install **`jaxlint[lsp]`**, then run **`jaxlint-lsp`** or **`python -m jaxlint.lsp`** (stdio; diagnostics-only v1). See **[docs/lsp.md](docs/lsp.md)**.

## Testing

```bash
uv sync --all-groups --extra docs --extra mcp
uv run pytest -q -m smoke   # CI fast path
uv run pytest -q -m mcp     # MCP server + tool helpers
uv run pytest -q           # full suite
uv run pytest -q --cov=jaxlint --cov-report=term-missing   # mirrors main-branch coverage step
```

Markers include `smoke`, `unit`, `integration`, `slow`, and `needs_jax`. **Optional `jaxlint[mcp]`** installs the MCP SDK; run **`jaxlint-mcp`** or **`python -m jaxlint.mcp`** (stdio) for agent-facing tools, and **`pytest -q -m mcp`** when developing them. **`jaxlint_check`** defaults to the legacy behaviour (pretty-printed JSON array of diagnostics, or `error:` strings). Set **`structured=true`** for a **`jaxlint.mcp.v1`** envelope (`success`, `diagnostics`, optional `error`, `meta` with `elapsed_ms`, `paths_resolved`, and requested `jobs` / `cache_dir`). Path mode can pass **`jobs`** and **`cache_dir`** through to **`run_checks`** (same semantics as **`jaxlint check -j`** / **`--cache-dir`**); these options are rejected for in-memory **`source`** checks. **`jaxlint_capabilities`** returns JSON describing the schema version, package version, and tool metadata. See [docs/contributing.md](docs/contributing.md) for workflow details, coverage commands, and CI expectations.

## Roadmap

The living plan (phased **near / medium / long-term**, plus what is already shipped) is **[docs/roadmap.md](docs/roadmap.md)**. **Release notes:** **[docs/changelog.md](docs/changelog.md)**. Highlights: **diagnostics-only LSP v1** ships as **`jaxlint[lsp]`** (see **[docs/lsp.md](docs/lsp.md)**); **MCP** agent-grade v1 ships as **`jaxlint[mcp]`** (`jaxlint-mcp`, **`jaxlint_capabilities`**, structured **`jaxlint_check`**, **`pytest -m mcp`**). Shipped / near-term work includes **SARIF** and **compact** CLI output, **autodoc** API pages, deeper **HLO** tests — CI already runs a **required** **`needs_jax`** job (`jax-hlo`). **`jaxlint check`** supports **`-j` / `--jobs`** and an optional **`--cache-dir`** disk cache (see [Roadmap](docs/roadmap.md)).

## Configuration

Add to `pyproject.toml`:

```toml
[tool.jaxlint]
select = ["JL", "JD", "JM"]
ignore = []
docstring-style = "numpy"
required-sections = ["Parameters", "Returns"]
strict-shapes = false
rubric-threshold = 0.75

[tool.jaxlint.per-file-ignores]
"tests/**" = ["JL005"]
```

See [docs/rules.md](docs/rules.md) for rule catalogue.

## Library

`jaxlint.core` does not import JAX. Use `jaxlint.hlo` (extra `hlo`) for `SemanticAnalyzer` and `GraphAuditSensor`.
