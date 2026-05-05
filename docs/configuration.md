# Configuration (`[tool.jaxlint]`)

jaxlint reads optional settings from the **first** `pyproject.toml` found when walking upward from:

- the `--config` directory (CLI), or
- `config_start` passed to `load_config` / `run_checks`, or
- the current working directory.

Parsed keys live in `LintConfig` (`jaxlint.core.config`). TOML keys use **kebab-case**; Python attributes use **snake_case**.

## Keys

| TOML key | Type | Default | Meaning |
|----------|------|---------|---------|
| `select` | list of strings | *(empty)* | If non-empty, only rules whose id matches **any** pattern (see below) run. |
| `ignore` | list of strings | *(empty)* | Rules matching any pattern are skipped after per-file rules. |
| `docstring-style` | string | `"numpy"` | Passed to the docstring style validator (Griffe-backed). |
| `required-sections` | list of strings | `["Parameters", "Returns"]` | Section titles required in public APIs (see JD002). |
| `section-order` | list of strings | `Parameters`, `Returns`, `Raises`, `Examples`, `Pseudocode`, `Hardware Notes` | Canonical order for JD003 warnings. |
| `strict-shapes` | bool | `false` | If `true`, JL005 is promoted to **error** severity. |
| `rubric-threshold` | float | `0.75` | Default threshold for `jaxlint score`. |
| `per-file-ignores` | table | `{}` | Map of **glob patterns** → rule patterns to disable for matching paths. |

### `select`, `ignore`, and `per-file-ignores` patterns

Matching is implemented by `pattern_matches` in `jaxlint.core.config`:

- Exact: `JL001` matches only `JL001`.
- Prefix: a pattern ending with `*` matches rule ids that **start with** the prefix before `*`. Example: `JL` matches every id starting with `JL`; `JL*` does the same.

`per-file-ignores` keys are matched with `fnmatch` against the **string form** of the file path passed to `rule_enabled` (typically relative paths like `tests/fixtures/foo.py`). Values can be a **single string** or a **list of strings**; each entry uses the same pattern rules as `ignore`.

## Examples

**Doc-only rules in CI:**

```toml
[tool.jaxlint]
select = ["JD", "JM"]
```

**Strict shape checks (JL005 as errors):**

```toml
[tool.jaxlint]
strict-shapes = true
```

**Ignore JL005 everywhere except tune per directory:**

```toml
[tool.jaxlint]
ignore = ["JL005"]

[tool.jaxlint.per-file-ignores]
"experimental/**" = ["JL012"]
"kernels/**/*.py" = ["JL005", "JL010"]
```

**Custom section order for JD003:**

```toml
[tool.jaxlint]
section-order = ["Parameters", "Returns", "Examples", "Raises"]
```
