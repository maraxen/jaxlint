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

## Documentation (Sphinx + Read the Docs)

Jaxlint tracks upstream JAX’s docs stack at a lighter weight: **Sphinx**, **MyST** (`myst-parser`), **`sphinx-book-theme`**, pinned in the **`docs`** optional extra of [`pyproject.toml`](../pyproject.toml).

Local HTML:

```bash
uv sync --extra docs
uv run sphinx-build -W -b html docs docs/_build/html
```

Link validation (respects [`linkcheck_ignore` in `docs/conf.py`](conf.py); trim that list once canonical URLs resolve):

```bash
uv run sphinx-build -b linkcheck docs docs/_build/linkcheck
```

**Read the Docs behavior**

- Builds are configured via [`.readthedocs.yml`](https://github.com/jax-ml/jaxlint/blob/main/.readthedocs.yml) at the repo root.
- **`formats: [pdf, htmlzip]`** produces versioned downloadable artifacts on **branch/tag builds**. Per RTD policy, pull-request preview builds typically emit **HTML only** — treat PDF/htmlzip verification as a **`latest`** / release build step.
- `sphinx.fail_on_warning` stays **`false`** in RTD YAML until warnings are extinguished (`true` tightens CI-style doc builds).

After the hosted project exists, update repository URLs in **`docs/conf.py`** and [`index.md`](index.md) if yours differ from the `jax-ml/jaxlint` placeholders.

## Documentation audit checklist (releases)

Before enabling stricter Sphinx settings (`fail_on_warning: true`) or shrinking `linkcheck_ignore`:

1. Hidden `{toctree}` in [`index.md`](index.md) lists every authored `.md` page.
2. `sphinx-build -b linkcheck` is clean aside from deliberate ignores or known external outages.
3. [`docs/cli.md`](cli.md) remains aligned with `jaxlint.cli.main` (spot-check flags).
4. An RTD **non-PR** build shows **HTML**, **PDF**, and **htmlzip** successes for `latest` (or your release alias).

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

GitHub Actions (see [`.github/workflows/ci.yml`](https://github.com/jax-ml/jaxlint/blob/main/.github/workflows/ci.yml)) runs Ruff plus smoke tests on every PR, a Sphinx HTML + linkcheck job, plus (on pushes to `main`) the full pytest suite with coverage uploads.
