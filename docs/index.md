# jaxlint documentation

jaxlint is a **CLI and Python library** for JAX-oriented static checks (AST “sensors”), docstring structure, MathJax/LaTeX hygiene, and optional StableHLO-oriented helpers.

- **`jaxlint.core`** does **not** import JAX. Use it from build tools, pre-commit, or your own runners.
- **`jaxlint[hlo]`** adds **`jaxlint.hlo`** (requires JAX). Use it when you need export/HLO-level analysis.

## Hosted documentation

HTML, PDF, and htmlzip builds are published via [Read the Docs](https://jaxlint.readthedocs.io/).

## Contents

| Document | What it covers |
|----------|----------------|
| [CLI reference](cli.md) | Commands, flags, exit codes |
| [Configuration](configuration.md) | `[tool.jaxlint]` in `pyproject.toml` |
| [Library usage](library.md) | `run_checks`, diagnostics, formatting |
| [Rule catalogue](rules.md) | Rule IDs, severity, fixtures |
| [API reference](api.md) | Generated reference for `jaxlint.core` |
| [HLO extra](hlo.md) | Installing and importing `jaxlint.hlo` |
| [Roadmap](roadmap.md) | Phased product and technical direction |
| [Changelog](changelog.md) | Release notes and unreleased doc/test changes |
| [Contributing](contributing.md) | Tests, coverage, CI markers |

Start from the [project README](https://github.com/jax-ml/jaxlint/blob/main/README.md) (install and CLI quick start). If the repository URL changes, update links in this page and in `docs/conf.py`.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Guides

cli
configuration
library
api
rules
hlo
roadmap
changelog
contributing
```
