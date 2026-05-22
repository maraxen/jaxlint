# Implementation audit — Hosted docs + CI docs job (Phase 5)

**audit_id:** `audit-impl-hosted-docs-ci-2026-05-05`  
**scope:** `.github/workflows/ci.yml` (`docs-should-run`, `paths-filter`, `docs` job), `docs/roadmap.md` (canonical URL table), `docs/contributing.md` (hosted §, CI parity), `docs/conf.py` comments, `docs/changelog.md` unreleased CI bullets.

## Verdict: **APPROVE-with-nits**

### Strengths

- **Honest 404** reporting for canonical URLs avoids fake green linkcheck.
- **Frozen glob list** in workflow header aids audits.
- **sphinx-build -W** on HTML vs **linkcheck** without `-W` is documented consistently with workflow steps.

### Nits (non-blocking)

1. **Branch protection:** If **`docs`** is a required check, PRs that skip the job show **skipped** — confirm repo settings or a neutral success job.
2. **Third-party action:** `dorny/paths-filter@v3` — align with org pin/supply-chain policy.

### Request changes?

**No.**
