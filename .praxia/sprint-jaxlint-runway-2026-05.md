# Sprint — Release runway (MIT, 0.1.0, CI, governance, release.yml)

**task_id:** `jaxlint-runway-2026-05`
**Plan audit:** `audit-plan-jaxlint-runway-2026-05` → **PASS after remediation** (remediation folded below).

## Locked decisions

| Item | Value |
|------|--------|
| License | **MIT** (replace Apache-2.0) |
| Authors | `Marielle Russo` `<67157875+maraxen@users.noreply.github.com>` |
| Version | **0.1.0** (`Development Status :: 3 - Alpha` unchanged) |
| Prior semver | Changelog keeps historical **0.3.x / 0.4.x** sections; **0.1.0** opens with one paragraph explaining internal/pre-PyPI numbering below |

## Plan-auditor remediation (must implement)

1. **CI:** Replace separate `pytest -m smoke`, `-m lsp`, `-m mcp`, and conditional full run with **one** step per matrix cell:  
   `uv run pytest -q --cov=jaxlint --cov-report=term-missing --cov-report=xml`  
   after `uv sync --all-groups --extra lsp --extra mcp`. Keep **Ruff** before pytest. Keep **`jax-hlo`** job unchanged (still `needs_jax` only with `[hlo]` + pinned jax).
2. **`release.yml`:** `permissions: contents: read`, `id-token: write` (OIDC). Publish with `pypa/gh-action-pypi-publish@release/v1`; PyPI project **`jaxlint`** must match `pyproject.toml` `name`.
3. **`docs/changelog.md`:** At **## 0.1.0**, first bullets = license MIT + public-metadata runway + semver reset; **one sentence**: sections labeled 0.3.x–0.4.x below document pre-0.1.0 development and are not separate PyPI lines unless already published.
4. **`docs/runway.md`:** Update synthesis table row that still says Apache-2.0 → **MIT** (and brief note if needed).

## R1 — Identity, license, metadata

- Replace **`LICENSE`** with standard MIT text; copyright **2026 Marielle Russo** (or "jaxlint contributors" if you prefer collective — **use Marielle Russo** per author lock).
- **`pyproject.toml`:** `license = { text = "MIT" }`; classifier `License :: OSI Approved :: MIT License` (remove Apache classifier); `[project.urls]`:
  - Homepage = `https://github.com/maraxen/jaxlint`
  - Documentation = `https://jaxlint.readthedocs.io/`
  - Repository = `https://github.com/maraxen/jaxlint`
  - Issues = `https://github.com/maraxen/jaxlint/issues`
  - Changelog = `https://jaxlint.readthedocs.io/en/latest/changelog.html`
- Classifiers: add `Programming Language :: Python :: 3.11`, `Programming Language :: Python :: 3.12`, `Typing :: Typed`.
- `authors = [{ name = "Marielle Russo", email = "67157875+maraxen@users.noreply.github.com" }]`
- `version = "0.1.0"`; sync **`src/jaxlint/__init__.py`** `__version__`.
- **`README.md`:** CLI includes `jaxlint version`; MCP lists **`jaxlint_rules`**, **`jaxlint_version`**.
- **`docs/lsp.md`:** Footnote "as of …" → **0.1.0**.

## R2 — Governance

- Root **`CONTRIBUTING.md`** → short pointer to **`docs/contributing.md`**.
- **`SECURITY.md`** — supported versions **0.1.x**; contact email same as authors.
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant 2.1 (standard).
- **`.github/ISSUE_TEMPLATE/bug_report.md`** + **`feature_request.md`** (YAML frontmatter OK).
- **`.github/PULL_REQUEST_TEMPLATE.md`** — link contributing + tiny checklist.

## R3 — CI

- Edit **`.github/workflows/ci.yml`** per remediation §1.
- Align **`Upload coverage.xml`** to run whenever coverage XML is produced (every PR/push on lint-test).
- Update **`docs/roadmap.md`** sentence if it still claims "full pytest only on main" (match new CI).

## R4 — LSP smoke + release verification

- Add/extend test: subprocess **`jaxlint-lsp`** on PATH, JSON-RPC `initialize` → `initialized` → `textDocument/didOpen` on temp `.py` with known violation; assert `publishDiagnostics`; env **`JAXLINT_LSP_DEBOUNCE_MS=0`**; **`@pytest.mark.lsp`**.
- **`docs/contributing.md`:** subsection **Release verification** (build, twine check, venv install wheel, CLI smoke).
- **`[dependency-groups] dev`:** add **`build`**, **`twine`** (pinned loosely `>=...`).
- Optional **`scripts/release_verify.sh`** — shebang, `set -euo pipefail`.

## R5 — Roadmap table

- **`docs/roadmap.md`:** add **Release runway sprint** table (RR1–RR9) with Status (**in progress** / **blocked on public GitHub** as appropriate). Do **not** change canonical URL HTTP status to 200 without evidence.

## R6 — Release workflow

- **`.github/workflows/release.yml`:** `on.push.tags: 'v*'`; job uses uv, `python -m build`, `twine check dist/*`, `pypa/gh-action-pypi-publish@release/v1` with permissions above; comment: enable Trusted Publishing on PyPI for **`jaxlint`** ← **`maraxen/jaxlint`** repo.

## Verification (fixer must run before handoff)

```bash
uv run ruff check src tests
uv sync --all-groups --extra lsp --extra mcp --extra hlo
uv run pytest -q --cov=jaxlint --cov-report=term-missing
uv run pytest -q -m lsp
uv sync --extra docs && uv run sphinx-build -W -b html docs docs/_build/html
```

## Out of scope

Creating GitHub repo, RTD project, PyPI Trusted Publisher UI, OpenCode staging (R7).
