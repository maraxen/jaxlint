# Sprint artifact: May 2026 — Config evolution (severity & config surfaces)

**task_id:** `jaxlint-sprint-2026-05`  
**plan_id:** `plan-config-evolution-2026-05`  
**Plan-auditor:** NEEDS_WORK → **accepted** with frozen **P1 design locks** below (no coding until locks acknowledged).

## Objective

Ship configurable **per-rule severity overrides**, documented **SARIF / GitHub Checks** level mapping from effective severity, optional **`jaxlint.toml`** + **`--config` file path**, and cache fingerprint updates — **backward compatible** when overrides are absent.

## P1 design locks (frozen — implement exactly)

### Config discovery & precedence

1. **`load_config(start)`** walks from `start.resolve()` if `start` is a directory, else from `start.parent`, through **`[root] + root.parents`** (same shape as today: check current dir then parents).

2. At each directory **`d`**, in order:
   - If **`d / "jaxlint.toml"`** exists → parse **`[tool.jaxlint]`** from that file only and **return** (do not read `pyproject.toml` in the same directory).
   - Else if **`d / "pyproject.toml"`** exists → parse **`[tool.jaxlint]`** from it and **return**.
   - Else continue upward.

3. **Conflict:** If **`jaxlint.toml`** and **`pyproject.toml`** both exist in the **same** directory → **`ValueError`** at load time with a message naming the directory (no silent merge).

4. **`--config` CLI:**  
   - Argument may be a **directory** (current behaviour: use as `start` for walk).  
   - Or a **file** path to **`jaxlint.toml`** or **`pyproject.toml`** → parse that file’s **`[tool.jaxlint]`** directly (no upward walk). Resolve relative paths inside TOML (e.g. `cache-dir`) against **the directory containing the loaded config file**.

5. **CLI `--select` / `--ignore`:** Unchanged — applied **after** `load_config` via `dataclasses.replace` on `LintConfig` in `check`/`doc` as today. No new CLI flags for severity in this sprint.

### Severity pipeline & JL005 / strict-shapes

6. **Base severity** comes from sensors/rules (e.g. JL005: `ERROR` if `strict_shapes` else `WARNING` in `shape_diagnostics`).

7. **Overrides:** After all diagnostics for a file are collected and **`rule_enabled`** filtering applied, apply **`effective_severity(d: Diagnostic, cfg: LintConfig) -> Diagnostic`** (or equivalent) so **`severity-overrides`** in config **always win** over emitted severity including JL005 + strict-shapes. Unknown rule keys in `severity-overrides` → **parse-time error** (fail fast).

8. **Wildcards:** Override keys use the same **`pattern_matches`** semantics as **`select`/`ignore`** (prefix + `*`).

### Cache

9. **`_canonical_config_dict`** must include a **stable serialization** of severity overrides (sorted keys, normalized severities).

10. **`CACHE_FORMAT_VERSION`:** Bump **1 → 2** when overrides land so old cache entries are never reused across schema change.

### Exporters

11. **`format_sarif`** / **`format_github_annotations_json`** read **`Diagnostic.severity`** only after the effective-severity step — **no second mapping layer**. Minimum CF4: correct **`level` / annotation_level** for error/warning/info; SARIF `rules` resource block is **optional stretch** only if time permits.

### LSP / MCP parity

12. **`jaxlint.lsp`**: Continues to call **`load_config(path.parent)`**; no separate merge rules. After config loading changes, smoke-test LSP paths still resolve the same config as CLI for the same tree layout.

13. **`jaxlint.mcp`**: Uses **`load_config`** for path-based checks — same discovery.

## Deliverables

| ID | Deliverable |
|----|-------------|
| **CF1** | `LintConfig` + TOML **`severity-overrides`** under **`[tool.jaxlint]`** (values: `error` / `warning` / `info`). |
| **CF2** | `effective_severity` applied in runner pipeline once, shared by **`run_checks`** and **`diagnostics_for_python_source`**. |
| **CF3** | Cache: `_canonical_config_dict` + **`CACHE_FORMAT_VERSION = 2`**. |
| **CF4** | SARIF + GitHub JSON verified against overridden severities (tests). |
| **CF5** | **`jaxlint.toml`** discovery + **`--config` file** semantics per locks above. |
| **CF6** | Tests (`test_config`, `test_runner` or dedicated), **`docs/configuration.md`**, **`docs/cli.md`**, **`docs/changelog.md`**, **`starter_toml_snippet()`**, semver **minor bump** (e.g. **0.3.0**). |

## Verification

- `uv run ruff check src tests`
- `uv run pytest -q -m smoke`
- `uv run pytest -q`
- `uv sync --all-groups --extra lsp --extra mcp && uv run pytest -q -m lsp -m mcp` as applicable for touched paths

## Out of scope

- Rich-only polish; RTD URL hygiene; LSP formatting; new MCP tools
- CLI flags for per-invocation severity overrides (file-based only this sprint)
