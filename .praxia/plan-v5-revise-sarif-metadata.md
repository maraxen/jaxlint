# Plan v5–v6 — SARIF rule metadata

**task_id:** `jaxlint-roadmap-remaining-2026`  
**Replaces:** `plan-v4-final-sarif-0.4.0-2026`  
**Semver target:** **`0.4.1`** (Rich polish consumed **`0.4.0`**.)

---

## Revision v6 (plan-auditor remediation, 2026-05-05)

Resolves **NEEDS_WORK** on DoD vs unknown **`rule_id`**, **`ruleIndex`** expectations, **`LintConfig`** threading, and severity strategy.

### Resolved policies

1. **`driver.rules` vs unknown `rule_id`**

   Build **`runs[].tool.driver.rules`** as the ordered list of descriptors for **every distinct `ruleId` appearing in `results`**. For IDs present in the jaxlint catalogue, use catalogue short text / default severity. For **unknown** IDs (custom plugins, typos), append a **synthetic** descriptor per missing id (`shortDescription` e.g. `"Unknown or custom rule"`, stable **`id`** == `ruleId`). Then **`len(rules) >= 1`** iff **`len(results) >= 1`**, and DoD tests stay consistent.

2. **`ruleIndex`**

   SARIF 2.1.0 does **not** require **`ruleIndex`** when **`ruleId`** is set. jaxlint **always sets **`ruleIndex`** on each **`result`** to the index of that result’s **`ruleId`** in **`driver.rules`** (viewer interoperability).

3. **`LintConfig` threading**

   Signature target: **`format_sarif(diagnostics, *, tool_version: str, cfg: LintConfig | None = None)`**. **`format_diagnostics`** passes **`cfg`** when SARIF is invoked from CLI paths that already loaded config. **`library.md`** documents that programmatic SARIF with **`sarif-rule-docs-base`** requires passing **`cfg`** (or whatever helper loads effective config).

4. **Severity**

   - **`results[].level`** remains the **effective** severity (**`severity-overrides`** applied), unchanged from today.
   - **`rules[].defaultConfiguration`** / reporting descriptors document **catalogue default** severity where applicable; changelog states results win for enforcement.

### Behaviour matrix (updated)

| Scenario | Expected |
|----------|-----------|
| Known **`rule_id`** | Descriptor from catalogue; **`helpUri`** = **`sarif-rule-docs-base` + "/" + rule_id** when base set |
| Unknown **`rule_id`** | Synthetic descriptor row; same **`ruleIndex`** linkage |
| **`sarif-rule-docs-base`** unset | Omit **`helpUri`** on rules (or only **`informationUri`** on driver — document chosen UX in changelog) |

---

## Problem statement (v5)

v4 stub still referenced **0.4.0** for SARIF enrichment; release slot must move forward. Auditors need an explicit **implementation matrix** and **unknown rule ID** behaviour.

## Goals

1. **`tool.driver.rules[]`** — Populate SARIF **rules** array from the jaxlint rule catalogue (rule id, short text, optional **helpUri**).
2. **`ruleIndex`** — Each **result** references **`ruleId`** consistently; **jaxlint sets **`ruleIndex`** per v6 policy above.
3. **Configurable docs base** — Optional **`LintConfig`** / CLI field **`sarif-rule-docs-base`** (HTTPS base URL); append **`/{rule_id}`** or documented pattern for **helpUri**.
4. **Severity / effective severity** — Per **Revision v6** split between catalogue defaults in **`rules`** vs **`results[].level`**.

## Files (likely)

- `src/jaxlint/core/runner.py` — **`format_sarif(..., cfg=...)`**; **`format_diagnostics`** passes **`cfg`** for SARIF.
- `src/jaxlint/core/config.py` — optional **`sarif-rule-docs-base`** (+ HTTPS validation if enforced).
- **`docs/library.md`**, **`docs/cli.md`**, **`docs/configuration.md`** — programmatic vs CLI **`cfg`** for SARIF extras.
- `tests/test_runner.py` or **`test_sarif_rules.py`** — matrix rows as parametrized tests.

## Definition of done

- [ ] **`0.4.1`** version bump + changelog + roadmap line under SARIF / config.
- [ ] Test: non-empty diagnostics ⇒ **`driver.rules`** length equals **count of distinct `ruleId`** in results (≥ 1).
- [ ] Test: each **`result`** has **`ruleIndex`** matching **`rules`** ordering.
- [ ] Test: unknown synthetic **`rule_id`** still yields descriptor + index.
- [ ] Test: optional **`sarif-rule-docs-base`** **`helpUri`** composition.

## Non-goals

- GitHub Advanced Security submission packs.
- SARIF **subscriptions** / **notifications** objects.

## Packages

| ID | Deliverable |
|----|-------------|
| SF1 | Rule descriptors array |
| SF2 | **`ruleIndex`** / linkage |
| SF3 | **`sarif-rule-docs-base`** + threading from **`format_diagnostics`** |
| SF4 | Unknown-id policy + tests |
| SF5 | Severity documentation in rule vs result |
| SF6 | Pytest matrix |
| SF7 | Docs **`cli.md`** / **`configuration.md`** + changelog |
