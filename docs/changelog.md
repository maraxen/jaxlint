# Changelog

Notable changes to jaxlint are listed here. Versions match the `version` field in [`pyproject.toml`](https://github.com/jax-ml/jaxlint/blob/main/pyproject.toml) at release time.

## Unreleased

- **Docs:** Added this changelog (per [roadmap steering](roadmap.md)).
- **Testing:** Golden StableHLO snippets live under [`tests/fixtures/hlo/`](https://github.com/jax-ml/jaxlint/tree/main/tests/fixtures/hlo) for selected `needs_jax` tests. The merge-blocking **jax-hlo** CI job still installs **`jaxlint[hlo]`** and JAX; golden text reduces dependence on live `jax.export` output shapes for those cases.

## 0.2.0

Packaged release line aligned with the shipped CLI, rules engine, SARIF / compact / GitHub diagnostic formats, Sphinx docs (including API reference for `jaxlint.core`), and optional **`jaxlint[hlo]`** analysis with merge-blocking `needs_jax` coverage. See the **Shipped baseline** section in the [Roadmap](roadmap.md).
