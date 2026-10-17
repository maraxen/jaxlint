---
name: using-jaxlint
description: Use when running jaxlint checks, interpreting lint violations, fixing JL-rule errors, or integrating jaxlint into a JAX workflow.
when_to_use: 'Triggers: jaxlint, lint, jax rule.'
---
# using-jaxlint

Jaxlint is a JAX-oriented linter that catches performance anti-patterns, API misuse, and documentation quality issues.

## When to use

- After writing JAX code (especially kernels with `@jax.jit`, `@jax.grad`, etc.)
- When a JL-rule violation appears in IDE/CI
- Before committing JAX changes or code reviews
- To understand why a shape-polymorphism or dispatch pattern is flagged

## Quick start

**Check a single file or directory:**
```bash
jaxlint check path/to/file.py
# or
jaxlint check src/my_jax_module/
```

**Read output:** Default (text) format is `file:line:col [SEVERITY] rule_id — message`.
To see other formats (compact, JSON, GitHub annotations, SARIF, rich table), use `--format`.

**See all rules:**
```bash
jaxlint rules
```

**Output formats and CLI flags agents commonly need:**
```bash
# Formats
jaxlint check --format text path/to/file.py      # Default: path:line:col [SEVERITY] rule_id — message
jaxlint check --format compact path/to/file.py   # Compact: path:line:col: severity: message (rule_id)
jaxlint check --format json path/to/file.py      # JSON array of diagnostics
jaxlint check --format github path/to/file.py    # GitHub annotations format
jaxlint check --format rich path/to/file.py      # Colored table with Unicode

# Common options
jaxlint check --ignore JD,JM src/               # skip doc and math rules
jaxlint check --select JL src/                  # performance rules only
jaxlint check --jobs 4 src/                     # parallel workers (default: min(32, CPU count))
jaxlint check --cache-dir .cache/ src/          # enable caching
```

**Skip checks you don't need:**
```bash
jaxlint check --no-doc file.py   # skip docstring/MathJax checks
jaxlint check --no-perf file.py  # skip performance (AST) sensors only
```

## Core rules (high-frequency)

| Rule | Severity | What it catches | Quick fix |
|------|----------|-----------------|-----------|
| **JL001** | ERROR | `.at[].set()` result immediately subscripted (scatter+gather anti-pattern) | Remove the subscript after `.set()` or restructure |
| **JL002** | WARNING | Short function has `@jax.jit` — may block fusion if used as helper | Remove JIT from helper functions or inline them |
| **JL003** | ERROR | Python `for`/`while` traced inside `@jax.jit` or `@jax.grad` | Use `jax.lax.scan`, `fori_loop`, or move loop outside the kernel |
| **JL004** | ERROR | `print`/`open`/file I/O inside a JAX kernel | Move I/O outside the JIT boundary or use callbacks |
| **JL005** | WARNING | `.shape` or `jnp.shape` inside kernel (breaks shape polymorphism) | Use shape arguments passed in; use `--strict-shapes` to make it an error |
| **JL010** | WARNING | `jnp.float64` usage — prefer f32/bf16 unless double precision is required | Change to `jnp.float32` or `jnp.bfloat16` |
| **JL012** | WARNING | Division in JIT kernel without safe reciprocal | Add zero checks or use `jnp.where` for safe division |
| **JD001** | ERROR | Public function/class missing docstring | Add a docstring (no underscores in name → public) |
| **JD002** | ERROR | Docstring missing required section (e.g., Parameters, Returns) | Add the section; configure `required-sections` in config |
| **JD004** | WARNING | Docstring Parameters do not cover all function arguments | Document all parameters in the docstring |
| **JD005** | WARNING | Function has return annotation but lacks Returns section in docstring | Add Returns section to docstring |

For the full catalog: `jaxlint rules`. Each rule has a fixture in `tests/fixtures/`.

## MCP integration

If jaxlint-mcp is available (`jaxlint[mcp]` extra installed), use these tools in MCP context:

| Tool | Purpose |
|------|---------|
| **jaxlint_check** | Run checks on filesystem paths or in-memory Python source; returns JSON diagnostics (supports `structured=true` for versioned envelope) |
| **jaxlint_rules** | List rule IDs with severity and description (same as `jaxlint rules` CLI) |
| **jaxlint_capabilities** | Return schema version, package version, and registered tool metadata |
| **jaxlint_version** | Return package version string |

Agents in MCP context can call these directly instead of shelling out to CLI. The `structured=true` mode on `jaxlint_check` returns metadata (elapsed time, paths resolved) and error detail.

## LSP

Start the language server with `jaxlint-lsp` or `python -m jaxlint.lsp` (requires `jaxlint[lsp]` extra). It publishes diagnostics for any open `.py` file over stdio; wire your editor to launch it. Diagnostics refresh with a 200 ms debounce on file changes.

**Configuration:** The server discovers `jaxlint.toml` or `[tool.jaxlint]` in `pyproject.toml` from the file's directory upward — same as CLI.

**No LSP code actions or formatting yet** — diagnostics only in v1.

See `docs/lsp.md` for setup details and `JAXLINT_LSP_DEBOUNCE_MS` env var to adjust timing.

## Configuration

**Suppress rules via configuration** (`jaxlint.toml` or `[tool.jaxlint]` in `pyproject.toml`):
```toml
[tool.jaxlint]
select = ["JL", "JD"]           # rules to enable (prefixes; default: all)
ignore = ["JL005", "JD003"]     # rules to ignore (can also use CLI --ignore JL005,JD003)

[tool.jaxlint.per_file_ignores]
"src/legacy/*.py" = ["JL001", "JL003"]  # ignore rules for legacy code
"tests/**" = ["JD001"]                  # skip docstring requirement in tests

[tool.jaxlint]
strict-shapes = false           # JL005 as error instead of warning
required-sections = ["Parameters", "Returns"]  # for JD002
section-order = ["Summary", "Parameters", "Returns", "Raises"]  # for JD003
docstring-style = "numpy"       # or "google", "sphinx"
cache-dir = ".jaxlint-cache"    # on-disk diagnostic cache (speeds up repeated runs)
```

See `docs/configuration.md` for all options and defaults.

## Going deeper

- **JAX fundamentals** → `using-jax` skill
- **Full rule catalog** → Run `jaxlint rules` or see docs/rules.md in the repo
- **Configuration reference** → `jaxlint init` (prints starter config) or `docs/configuration.md`
- **LSP setup by editor** → `docs/lsp.md`
- **Library API (not CLI)** → `docs/library.md` or `docs/api.md`
- **Other commands** → `jaxlint doc` (docstring-only checks), `jaxlint score` (rubric scoring), `jaxlint hlo` (HLO analysis), `jaxlint init` (create config)
