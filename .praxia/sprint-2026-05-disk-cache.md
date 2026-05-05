# Sprint artifact: May 2026 — P09 Disk cache (`--cache-dir`)

**task_id:** `jaxlint-sprint-2026-05-p09-cache`  
**Plan:** `plan-p09-disk-cache-2026-05`  
**Plan-auditor:** NEEDS_WORK → **accepted** with explicit resolutions below.

## Theme

Ship **opt-in on-disk caching** of per-file **`Diagnostic`** JSON for `jaxlint check`, with config fingerprint + mtime/size invalidation, atomic writes, and POSIX **flock** on writes; fail-open on I/O or corrupt entries.

## Plan-auditor resolutions (gates)

| Concern | Resolution |
|---------|------------|
| CLI vs TOML | **`--cache-dir` overrides** `[tool.jaxlint] cache-dir` when the CLI flag is passed; otherwise use config value; both absent ⇒ no cache. |
| P09e default | **POSIX:** `fcntl.flock` exclusive lock on `*.lock` adjacent to cache blob during write; shared lock optional on read (validate JSON → miss if corrupt). **Windows:** atomic `os.replace` only; documented best-effort for concurrent writers. |
| Fail-open | Unreadable cache, corrupt JSON, permission errors ⇒ **miss**, run `_diagnostics_for_file`, attempt store. |
| Hit/miss telemetry | **Out of scope** for P09. |
| Early tests | Contract tests land in **P09f** with the implementation (acceptable for this sprint). |

## Task IDs

| ID | Deliverable |
|----|-------------|
| **P09a** | `--cache-dir` on `check`; `run_checks(..., cache_dir=)` |
| **P09b** | `jaxlint.core.cache`: format version, fingerprint, JSON envelope, atomic store + flock |
| **P09c** | Runner wraps per-file work with get/put |
| **P09d** | `[tool.jaxlint] cache-dir` (relative to `pyproject.toml` directory); CLI wins |
| **P09e** | Locking as above |
| **P09f** | Tests: warm==cold, parallel+jobs, invalidation, fingerprint, corrupt blob, CLI |

## Verification

- `uv run ruff check src tests`
- `uv run pytest -q -m smoke`
- `uv run pytest -q`

## Out of scope

`doc` / `score` / `hlo` caching; LSP; AST cache; cache statistics.
