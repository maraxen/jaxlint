# Contributing to jaxlint

## Environment

```bash
cd /path/to/jaxlint
uv sync --all-groups --extra docs
```

Development dependencies live in `[dependency-groups] dev`. Sphinx / Read the Docs tooling uses the **`docs`** optional extra (`[project.optional-dependencies]` in `pyproject.toml`).

## Linting

```bash
uv run ruff check src tests
```

## Running tests

| Command | Purpose |
|---------|---------|
| `uv run pytest -q -m smoke` | Fast PR-style checks Typer/import paths |
| `uv run pytest -q -m unit` | Config helpers, serializers, narrowly scoped logic |
| `uv run pytest -q -m integration` | `run_checks` + CLI integration |
| `uv run pytest -q` | Entire suite (includes optional skips) |

`jax`-dependent checks use `@pytest.mark.needs_jax` and importorskip—they are skipped unless JAX is installed (for example via `uv sync --extra hlo`).

**Language server (`jaxlint[lsp]`)** — install with `uv sync --all-groups --extra lsp` (or `--extra lsp` alone). LSP-focused tests use **`@pytest.mark.lsp`** (see `tests/test_lsp_stdio.py`); CI also runs `tests/test_lsp_convert.py` and `tests/test_diagnostics_for_python_source.py` by path. Remaining **non-blocking** follow-ups for v1 are listed under **Open TODOs (LSP v1 reviewer nits)** in [Roadmap — November 2026 LSP sprint](roadmap.md#open-todos-lsp-v1-reviewer-nits).

### HLO golden fixtures

Some **`needs_jax`** tests load **pinned StableHLO / MLIR text** from `tests/fixtures/hlo/` instead of calling `jax.export` for every assertion. That lowers churn when export formatting changes, but **does not** remove the JAX + `jaxlint[hlo]` requirement: `SemanticAnalyzer` and the test module still import JAX-backed modules.

When adding a golden file, keep it **minimal**, name it after the behaviour under test (for example `outfeed_side_effect.mlir`), and wire a `FakeExported`-style duck type or parser entry point in `tests/test_hlo_semantics.py` (or a dedicated test module) so CI’s `jax-hlo` job exercises it.

## Documentation (Sphinx + Read the Docs)

Jaxlint tracks upstream JAX’s docs stack at a lighter weight: **Sphinx**, **MyST** (`myst-parser`), **`sphinx-book-theme`**, pinned in the **`docs`** optional extra of [`pyproject.toml`](../pyproject.toml).

Local HTML:

```bash
uv sync --extra docs
uv run sphinx-build -W -b html docs docs/_build/html
```

Link validation (see [`linkcheck_ignore` in `docs/conf.py`](conf.py); ignores cover the canonical Read the Docs and `jax-ml/jaxlint` GitHub URLs until linkcheck can verify them reliably):

```bash
uv run sphinx-build -b linkcheck docs docs/_build/linkcheck
```

**Read the Docs behavior**

- Builds are configured via [`.readthedocs.yml`](https://github.com/jax-ml/jaxlint/blob/main/.readthedocs.yml) at the repo root.
- **`formats: [pdf, htmlzip]`** produces versioned downloadable artifacts on **branch/tag builds**. Per RTD policy, pull-request preview builds typically emit **HTML only** — treat PDF/htmlzip verification as a **`latest`** / release build step.
- `sphinx.fail_on_warning` is **`true`** in the repo-root `.readthedocs.yml`, so RTD builds fail on Sphinx warnings once the hosted pipeline matches local `sphinx-build -W` hygiene.

After the hosted project exists, update repository URLs in **`docs/conf.py`** and [`index.md`](index.md) if yours differ from the `jax-ml/jaxlint` placeholders.

## Documentation audit checklist (releases)

Maintainers should periodically verify (especially before a release):

1. Hidden `{toctree}` in [`index.md`](index.md) lists every authored `.md` page (including [`changelog.md`](changelog.md)).
2. `sphinx-build -b linkcheck` is clean aside from deliberate ignores or known external outages.
3. [`docs/cli.md`](cli.md) remains aligned with `jaxlint.cli.main` (spot-check flags).
4. An RTD **non-PR** build shows **HTML**, **PDF**, and **htmlzip** successes for `latest` (or your release alias).
5. Shrink **`linkcheck_ignore`** in `docs/conf.py` when canonical RTD/GitHub URLs become reliable for automated linkcheck.

RTD already sets **`sphinx.fail_on_warning: true`**; local `sphinx-build -W -b html` should stay clean.

## Coverage

Generate terminal + HTML reports:

```bash
uv run pytest -q \
  --cov=jaxlint \
  --cov-report=term-missing \
  --cov-report=html:htmlcov
```

Open `htmlcov/index.html` locally after the run completes to browse annotated source.

**Baseline snapshot (manual, May 2026):** `uv run pytest -q --cov=jaxlint` reports **~76% overall line coverage** with `needs_jax` tests skipped locally. The largest remaining gaps are `jaxlint.hlo` (needs JAX plus export plumbing) and some doc-style edge cases.

[`pyproject.toml`](https://github.com/jax-ml/jaxlint/blob/main/pyproject.toml) configures `[tool.coverage.run]` / `[tool.coverage.report]`, including a conservative **`fail_under`** guard for accidental regressions.

## Continuous integration

GitHub Actions (see [`.github/workflows/ci.yml`](https://github.com/jax-ml/jaxlint/blob/main/.github/workflows/ci.yml)) runs Ruff plus smoke tests on every PR, a dedicated **JAX / HLO** job that installs `jaxlint[hlo]` and runs `pytest -m needs_jax`, a Sphinx HTML + linkcheck job, plus (on pushes to `main`) the full pytest suite with coverage uploads from the main `lint-test` matrix.
