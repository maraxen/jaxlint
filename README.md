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

## Testing

```bash
uv sync --all-groups --extra docs
uv run pytest -q -m smoke   # CI fast path
uv run pytest -q           # full suite
uv run pytest -q --cov=jaxlint --cov-report=term-missing   # mirrors main-branch coverage step
```

Markers include `smoke`, `unit`, `integration`, `slow`, and `needs_jax`. See [docs/contributing.md](docs/contributing.md) for workflow details, coverage commands, and CI expectations.

## Roadmap

The living plan (phased **near / medium / long-term**, plus what is already shipped) is **[docs/roadmap.md](docs/roadmap.md)**. Highlights: **LSP** and **MCP** remain planned integrations; nearer work includes richer **CLI output**, **optional JAX CI**, **autodoc**, and harder **HLO** tests.

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
