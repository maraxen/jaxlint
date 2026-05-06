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
| `uv run pytest -q -m smoke` | Fast Typer/import subset (`smoke` marker) |
| `uv run pytest -q -m unit` | Config helpers, serializers, narrowly scoped logic |
| `uv run pytest -q -m integration` | `run_checks` + CLI integration |
| `uv run pytest -q` | Entire suite (includes optional skips) |
| `uv run pytest -q --cov=jaxlint --cov-report=term-missing` | Same coverage gate as CI `lint-test` |
| `uv run pytest -q -m mcp` | MCP tool helpers (`jaxlint[mcp]` required) |
| `uv run pytest -q -m lsp` | Language server — run when touching **`jaxlint/lsp/`** (same marker as CI LSP tests; install `jaxlint[lsp]`) |

`jax`-dependent checks use `@pytest.mark.needs_jax` and importorskip—they are skipped unless JAX is installed (for example via `uv sync --extra hlo`).

- When you change code under **`jaxlint/lsp/`**, run **`uv run pytest -q -m lsp`** so local results match CI’s LSP-marked tests.

**Language server (`jaxlint[lsp]`)** — install with `uv sync --all-groups --extra lsp` (or `--extra lsp` alone). LSP-marked tests cover convert helpers, in-memory runner parity, and stdio JSON-RPC integration. Remaining **non-blocking** follow-ups for v1 are listed under **Open TODOs (LSP v1 reviewer nits)** in [Roadmap — November 2026 LSP sprint](roadmap.md#open-todos-lsp-v1-reviewer-nits).

### HLO golden fixtures

See **[Goldens: mapping vs counts](hlo.md#goldens-mapping-vs-counts)** on the [HLO extra](hlo.md) page for how **`get_source_mapping`**, **`get_operation_inventory`**, and **`get_operations`** relate to golden assertions.

Some **`needs_jax`** tests load **pinned StableHLO / MLIR text** from `tests/fixtures/hlo/` instead of calling `jax.export` for every assertion. That lowers churn when export formatting changes, but **does not** remove the JAX + `jaxlint[hlo]` requirement: `SemanticAnalyzer` and the test module still import JAX-backed modules.

When adding a golden file, keep it **minimal**, name it after the behaviour under test (for example `outfeed_side_effect.mlir`), and wire a `FakeExported`-style duck type or parser entry point in `tests/test_hlo_semantics.py` (or a dedicated test module) so CI’s `jax-hlo` job exercises it.

## Documentation (Sphinx + Read the Docs)

Jaxlint tracks upstream JAX’s docs stack at a lighter weight: **Sphinx**, **MyST** (`myst-parser`), **`sphinx-book-theme`**, pinned in the **`docs`** optional extra of [`pyproject.toml`](../pyproject.toml).

### Hosted documentation & links

Canonical **Read the Docs** and **GitHub** URLs, measured **HTTP status** (including **404** where that is still true), **`linkcheck_ignore`** policy, and **next review** date live in the **[Canonical hosted URLs](roadmap.md#canonical-hosted-urls)** table on the [Roadmap](roadmap.md). The ignore regexes and inline evidence are in **[`docs/conf.py`](conf.py)** (`linkcheck_ignore`).

Local HTML:

```bash
uv sync --extra docs
uv run sphinx-build -W -b html docs docs/_build/html
```

Link validation (see [`linkcheck_ignore` in `docs/conf.py`](conf.py); ignores cover the canonical Read the Docs and `maraxen/jaxlint` GitHub URLs until linkcheck can verify them reliably):

```bash
uv run sphinx-build -b linkcheck docs docs/_build/linkcheck
```

**`sphinx-build -W` vs `linkcheck`**

- **HTML / RTD:** Local CI and Read the Docs use **warnings-as-errors** for the main documentation build: **`sphinx-build -W -b html ...`** in [`.github/workflows/ci.yml`](https://github.com/maraxen/jaxlint/blob/main/.github/workflows/ci.yml) and **`sphinx: fail_on_warning: true`** in [`.readthedocs.yml`](../.readthedocs.yml). Any Sphinx warning fails that build.
- **`linkcheck`:** The same CI job runs **`sphinx-build -b linkcheck`** **without** **`-W`**. That means generic Sphinx warnings are **not** promoted to errors by `-W` on the linkcheck step; **`linkcheck`** still exits non-zero when it reports broken links that are **not** covered by **`linkcheck_ignore`**.

**Read the Docs behavior**

- Builds are configured via [`.readthedocs.yml`](https://github.com/maraxen/jaxlint/blob/main/.readthedocs.yml) at the repo root.
- **`formats: [pdf, htmlzip]`** produces versioned downloadable artifacts on **branch/tag builds**. Per RTD policy, pull-request preview builds typically emit **HTML only** — treat PDF/htmlzip verification as a **`latest`** / release build step.
- `sphinx.fail_on_warning` is **`true`** in the repo-root `.readthedocs.yml`, so RTD builds fail on Sphinx warnings once the hosted pipeline matches local `sphinx-build -W` hygiene.

After the hosted project exists, update repository URLs in **`docs/conf.py`** and [`index.md`](index.md) if yours differ from the `maraxen/jaxlint` placeholders.

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

[`pyproject.toml`](https://github.com/maraxen/jaxlint/blob/main/pyproject.toml) configures `[tool.coverage.run]` / `[tool.coverage.report]`, including a conservative **`fail_under`** guard for accidental regressions.

## Release verification

Before tagging a release, smoke-test artifacts the way consumers will install them:

```bash
uv sync --all-groups
uv run python -m build
uv run twine check dist/*
rm -rf .release-venv && uv venv .release-venv && source .release-venv/bin/activate
uv pip install "jaxlint[lsp,mcp]" --find-links dist --no-index --upgrade
jaxlint version
jaxlint-lsp --help
jaxlint-mcp --help
```

Optional helper (same sequence): [`scripts/release_verify.sh`](../scripts/release_verify.sh).

## Continuous integration

GitHub Actions (see [`.github/workflows/ci.yml`](https://github.com/maraxen/jaxlint/blob/main/.github/workflows/ci.yml)) runs **Ruff**, **full pytest with coverage** on every PR and push (`lint-test` matrix with **`uv sync --all-groups --extra lsp --extra mcp`**), a dedicated **JAX / HLO** job that installs `jaxlint[hlo]` and runs `pytest -m needs_jax`, and (on pushes to `main`/`master` or docs-touched PRs) Sphinx HTML + linkcheck. On **pull requests**, that docs job runs only when **docs-related paths** change (see the path-globs comment at the top of the workflow file).
