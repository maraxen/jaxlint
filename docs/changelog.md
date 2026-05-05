# Changelog

Notable changes to jaxlint are listed here. Versions match the `version` field in [`pyproject.toml`](https://github.com/jax-ml/jaxlint/blob/main/pyproject.toml) at release time.

## Unreleased

- **Docs:** Refreshed **`linkcheck_ignore`** rationale in `docs/conf.py` with **`curl -sI -L`** status evidence (2026-05-05); next scheduled review **2026-10-01** while RTD/GitHub URLs remain **404**.
- **Testing:** Golden StableHLO under [`tests/fixtures/hlo/`](https://github.com/jax-ml/jaxlint/tree/main/tests/fixtures/hlo): outfeed / **`stablehlo.while`** **`audit_exported`** paths, plus **`add_constant_kernel`** / **`multiply_reshape_kernel`** inventory goldens (duck-typed fake exports; **`needs_jax`**). Merge-blocking **jax-hlo** CI installs **`jaxlint[hlo]`** and JAX.
- **CI:** The **`jax-hlo`** workflow pins **`jax==0.10.0`** and **`jaxlib==0.10.0`** after `uv sync` so **`needs_jax`** runs stay reproducible.
- **CLI:** `jaxlint check` and `jaxlint doc` accept **`--format rich`** (Rich table: severity, rule, location, message). Depends on explicit **`rich`** (see `pyproject.toml`).
- **Docs:** Added this changelog (per [roadmap steering](roadmap.md)).

## 0.2.0

Packaged release line aligned with the shipped CLI, rules engine, SARIF / compact / GitHub diagnostic formats, Sphinx docs (including API reference for `jaxlint.core`), and optional **`jaxlint[hlo]`** analysis with merge-blocking `needs_jax` coverage. See the **Shipped baseline** section in the [Roadmap](roadmap.md).
