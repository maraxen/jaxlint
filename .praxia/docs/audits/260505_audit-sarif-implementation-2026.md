# Implementation audit — SARIF 0.4.1 (`format_sarif` + config)

**audit_id:** `audit-sarif-impl-pre-commit-2026-05-05`  
**scope:** `src/jaxlint/core/runner.py` (`format_sarif`, `_RULE_CATALOGUE` from `RULES`), `LintConfig.sarif_rule_docs_base`, CLI threading, `tests/test_runner.py` / `tests/test_config.py`.

## Verdict: **APPROVE**

### Strengths

- **Single catalogue source:** `_RULE_CATALOGUE` derives from `jaxlint.core.rules.RULES` — no duplicate strings vs `rules.py`.
- **Unknown `rule_id`:** Synthetic descriptor; **`results[].level`** uses diagnostic severity (effective pipeline preserved elsewhere).
- **`ruleIndex`:** Stable ordering from **first-seen** distinct ids in **`sort_key()`** order; tests cover alignment with **`JL001`/`JM003`** sample set.
- **`helpUri`:** Base trimmed; HTTPS validated at config parse.
- **Empty diagnostics:** **`rules`** omitted when no results — matches test expectations.

### Nits applied before commit

- **`format_diagnostics`** docstring documents optional **`cfg`** for SARIF **`helpUri`** (discoverability).

### Request changes?

**No.**
