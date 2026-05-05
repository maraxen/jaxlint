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

## CLI

- `jaxlint check [PATHS]...` — performance sensors (AST) plus doc section and MathJax checks by default
- `jaxlint doc [PATHS]...` — doc sections + MathJax only
- `jaxlint score [PATHS]...` — rubric scores for docstrings (optional `--jsonl`, `--baseline`)
- `jaxlint rules` — list rule IDs
- `jaxlint init` — print a starter `[tool.jaxlint]` block
- `jaxlint hlo` — hints for optional `jaxlint.hlo` (requires `--extra hlo`)

## Testing

```bash
uv sync --all-groups
uv run pytest -q -m smoke   # CI fast path
uv run pytest -q           # full suite
uv run pytest -q --cov=jaxlint --cov-report=term-missing   # mirrors main-branch coverage step
```

Markers include `smoke`, `unit`, `integration`, `slow`, and `needs_jax`.

## Roadmap

- **LSP**: not bundled; planned as a thin editor integration over `jaxlint.core` diagnostics.
- **MCP**: no MCP server in this repository; could wrap the same API separately if useful.

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
