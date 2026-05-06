# Release runway — `maraxen/jaxlint` first public cut

**task_id:** `jaxlint-release-readiness-2026-05-06`
**Status:** *Pre-publish*. Synthesized from a parallel critique/defense audit (see `.praxia/audits.jsonl` rows `audit-release-readiness-critique-2026-05-06` and `audit-release-readiness-defense-2026-05-06`).
**Synthesis verdict:** **GATE-with-runway** — proceed *along this runway*, not directly to `pypi upload`. The defense's own minimum checklist concedes most GATE blockers; the disagreement is **sequencing**, not direction.

## Why a runway, not a "ship now"

| Claim | Defense | Critique | Synthesis |
|-------|---------|----------|-----------|
| MIT + src layout + hatchling + 143/1 passing + 79.7% coverage | strength | acknowledged | **engineering is alpha-ready** |
| Missing `[project.urls]`, 3.11/3.12 classifiers, `py.typed` already in tree | "post-ship polish" | "PyPI metadata gap" | **fix before first upload** (cheap; defines the project page) |
| `authors=[{name="Marielle"}]` no email; `LICENSE` says "jaxlint contributors"; copyright 2026 | not raised | authenticity / supply-chain risk | **fix** — pick "single author" or "contributors" and be consistent |
| No `SECURITY.md` / `CODE_OF_CONDUCT.md` / top-level `CONTRIBUTING.md` / issue+PR templates | not raised | required signals for first OSS release | **add** — minimal versions, link `docs/contributing.md` |
| PR CI doesn't run full suite + coverage gate | acceptable trade-off | correctness risk for first external user | **tighten** — full suite on PR (cost is small; current PRs are fast) |
| `jaxlint.lsp.server` 0% line coverage in parent | "subprocess artifact, integration tests pass" | "no end-to-end stdio smoke" | **add one stdio smoke test** spawning `jaxlint-lsp` and exchanging `initialize` / `didOpen` |
| RTD + `github.com/maraxen/jaxlint` 404 | "chicken-and-egg, resolves on push" | "README's first link is broken at publish time" | **stand up GitHub + RTD before PyPI upload** so first-impression links work |
| No release workflow, no tags, no `python -m build` / `twine check` smoke | not raised | supply-chain | **add `.github/workflows/release.yml`** with PyPI Trusted Publishing |
| Version `0.4.1` + `Development Status :: 3 - Alpha` | "honest signaling" | "semver/lifecycle mismatch — 0.4.x reads as ruff-style stable-ish" | **either reset to `0.1.0` or graduate to Beta**; do not ship the contradiction |
| OpenCode staging | "real users get value today" | "theoretical until exercised through MCP Inspector" | **stage post-publish** and treat as 0.5 milestone evidence |

GATE wins where the two disagree (workspace audit-synthesis convention). The defense is right that the *engineering* is ready; the critique is right that the *release* is not.

## Phased runway

Each phase has a **gate** (single objective question). Don't advance until the gate clears.

### Phase R1 — Identity & metadata (≤ 1 day)

**Gate:** *Does the PyPI page render correctly with valid metadata, and does the LICENSE match reality?*

- Decide: **single-author "Marielle <email>"** or **"jaxlint contributors"** with at least one external acknowledgement. Update `pyproject.toml` `authors` and `LICENSE` copyright line consistently.
- Add `[project.urls]` to `pyproject.toml`:
  - `Homepage = "https://github.com/maraxen/jaxlint"`
  - `Documentation = "https://jaxlint.readthedocs.io/"`
  - `Repository = "https://github.com/maraxen/jaxlint"`
  - `Issues = "https://github.com/maraxen/jaxlint/issues"`
  - `Changelog = "https://jaxlint.readthedocs.io/en/latest/changelog.html"`
- Add classifiers: `Programming Language :: Python :: 3.11`, `:: 3.12`, `Typing :: Typed` (the `src/jaxlint/py.typed` marker is already present).
- Decide version semantics:
  - **Option A — reset to `0.1.0`** (matches `Alpha`). Re-tag changelog history as `0.1.0` with a "previously developed under 0.x.y" note.
  - **Option B — graduate to `Development Status :: 4 - Beta`** and keep `0.4.x`.
  - The runway assumes **Option A** unless the maintainer explicitly chooses B.
- Refresh `docs/lsp.md` ("as of 0.4.0"), README CLI list (add `version`), and README MCP paragraph (add `jaxlint_rules` / `jaxlint_version`). README's `--format` line stays as-is; optionally mention `github-annotations` alias.

### Phase R2 — Governance & community files (≤ 1 day)

**Gate:** *Does GitHub's "community standards" checklist show green for a first-time visitor?*

- Top-level `CONTRIBUTING.md` (one screen; can simply summarize and link `docs/contributing.md`).
- `SECURITY.md` (single contact + supported-versions table — even if it's only `0.1.x`).
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1 verbatim is fine).
- `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md` (minimal frontmatter).
- `.github/PULL_REQUEST_TEMPLATE.md` (link contributing, DoD checklist).

### Phase R3 — CI tightening (≤ 1 day)

**Gate:** *Can a first external PR break a `unit` / `integration` / cache test without CI catching it?*

- `.github/workflows/ci.yml`: run `uv run pytest -q --cov=jaxlint --cov-report=term-missing` on **every** PR (drop the `main`-only guard) on the existing 3.11/3.12 matrix. Coverage runtime is currently ~7s with full extras — cost is negligible.
- Keep the merge-blocking `jax-hlo` job as-is.
- Add a `release` job stub (no-op until R5) so the matrix is visible.

### Phase R4 — LSP smoke & build verification (≤ 1 day)

**Gate:** *Have we exercised the actual published artifacts end-to-end?*

- One **subprocess stdio smoke test** for the LSP: spawn `jaxlint-lsp`, send `initialize` + `initialized` + `textDocument/didOpen`, assert a `textDocument/publishDiagnostics` arrives for a JL-violating snippet. Mark `@pytest.mark.lsp` so it joins the existing PR job.
- Add a **build-and-install smoke** (local script or CI job): `python -m build` → `twine check dist/*` → install the wheel into a fresh venv → `jaxlint --version`, `jaxlint-lsp --help`, `jaxlint-mcp --help` all exit 0. Document the command sequence in `docs/contributing.md` under a "release verification" subsection.

### Phase R5 — Public surfaces stand up (≤ 1 day)

**Gate:** *Do every link in the README and roadmap resolve to HTTP 200?*

- Create `github.com/maraxen/jaxlint`, push `main`, enable Issues + Discussions (optional but cheap).
- Import the repo in Read the Docs; verify the project slug is **`jaxlint`** (matches the README link); trigger a build.
- Once both return **200**, shrink `linkcheck_ignore` in `docs/conf.py` and update the **Canonical hosted URLs** table in `docs/roadmap.md` (`HTTP 200`, today's date, drop the `404` rationale).

### Phase R6 — Release workflow & first publish (≤ 1 day)

**Gate:** *Can we ship subsequent versions reproducibly from a tag without anyone touching an API token?*

- Configure **PyPI Trusted Publishing** for the repo (PyPI side: pending publisher; GitHub side: environment).
- Add `.github/workflows/release.yml`:
  - Trigger: annotated tag matching `v*`.
  - Steps: checkout → setup uv → `python -m build` → `twine check dist/*` → publish via Trusted Publishing → attach sdist+wheel to a GitHub Release whose body is the matching `docs/changelog.md` section.
- Cut `v0.1.0a1` (pre-release alpha) or `v0.1.0` when stable, or `v0.4.2` under Option B. Confirm the wheel + sdist on PyPI install cleanly in a fresh venv.

### Phase R7 — OpenCode staging (post-publish, ≤ 1 day)

**Gate:** *Have we observed a real MCP host calling `jaxlint_check` on a real path with structured output, and a real editor with the OpenCode `lsp.jaxlint` block reporting diagnostics?*

Follows the staging plan from the orchestrator's prior recon (`docs/lsp.md`, README MCP section, OpenCode config schema).

1. `pip install jaxlint[lsp,mcp]` from PyPI in a clean venv.
2. Clone a small JAX-ish sample repo; drop in a minimal `jaxlint.toml`.
3. CLI baseline: `jaxlint check .` — capture diagnostics output for comparison.
4. **MCP Inspector pass:** run `jaxlint-mcp`; confirm tool list (`jaxlint_check`, `jaxlint_capabilities`, `jaxlint_rules`, `jaxlint_version`); call `jaxlint_check` with `structured=true` on the same paths; diff the diagnostics against step 3.
5. **OpenCode MCP block:** add `mcp.jaxlint` (local stdio) to OpenCode config; invoke from an agent prompt; confirm parity with step 4.
6. **OpenCode LSP block:** add `lsp.jaxlint` with `extensions: [".py"]`; open a `.py` file with a known JL violation; confirm diagnostics track edits. Set `JAXLINT_LSP_DEBOUNCE_MS=0` while debugging.
7. **Conflict check:** if pyright also attaches to `.py`, narrow extensions or disable one server so behavior is predictable.
8. Capture a transcript / screenshot and attach to the next release notes (or a `docs/staging-opencode.md` note) — converts the OpenCode claim from theoretical to evidenced.

## Definition of Done (overall)

The runway is **complete** when:

- [ ] PyPI page for `jaxlint` shows Homepage / Docs / Repo / Issues / Changelog.
- [ ] `pip install jaxlint` in a clean venv exposes `jaxlint`, `jaxlint-lsp`, `jaxlint-mcp` and the right `--version`.
- [ ] `github.com/maraxen/jaxlint` and `jaxlint.readthedocs.io` both return **200**.
- [ ] Every PR runs the full pytest + coverage gate on 3.11/3.12, plus `jax-hlo`.
- [ ] An LSP stdio smoke test asserts `publishDiagnostics` round-trip from a spawned `jaxlint-lsp`.
- [ ] `release.yml` publishes from `v*` tags via Trusted Publishing.
- [ ] Version classifier matches version: either `Alpha` + `0.1.x`, or `Beta` + `0.4.x` — not the current contradiction.
- [ ] An OpenCode session has actually exercised both `lsp.jaxlint` and `mcp.jaxlint`.

## Roadmap deltas

Move these into `docs/roadmap.md` as a new **"Release runway"** sprint section:

| ID | Deliverable | Phase |
|----|-------------|-------|
| RR1 | `[project.urls]` + 3.11/3.12 + `Typing :: Typed` classifiers; reconciled `authors` + `LICENSE` | R1 |
| RR2 | Version semantics decision (reset 0.1.0 or graduate Beta); doc/README/lsp version refresh | R1 |
| RR3 | `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue + PR templates | R2 |
| RR4 | Full pytest + coverage gate on every PR | R3 |
| RR5 | LSP stdio subprocess smoke test (`@pytest.mark.lsp`) | R4 |
| RR6 | `python -m build` + `twine check` + clean-venv install smoke (script + docs) | R4 |
| RR7 | GitHub repo public, RTD project built once; `linkcheck_ignore` shrunk; canonical URLs table updated | R5 |
| RR8 | `release.yml` with Trusted Publishing; **`v0.1.0a1`** (alpha) or **`v0.1.0`** (stable) tag and first PyPI release | R6 |
| RR9 | OpenCode staging transcript (MCP Inspector + LSP) attached to the release notes | R7 |

Existing **Polish horizon** items remain post-RR9 work.
