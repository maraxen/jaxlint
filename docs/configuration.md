# Configuration (`jaxlint.toml` / `pyproject.toml`)

jaxlint discovers optional settings by walking **upward** from a starting directory (`[root] + root.parents`):

- the `--config` directory (CLI), when `--config` points at a directory;
- the current working directory, when `--config` is omitted;
- or **only** the `--config` file, when `--config` points at `jaxlint.toml` or `pyproject.toml` (no upward walk).

At each directory during the walk:

1. If **`jaxlint.toml`** exists in that directory, parse configuration from it and stop (**`pyproject.toml` in the same directory is not read**).
2. Else if **`pyproject.toml`** exists there, parse **`[tool.jaxlint]`** from it and stop.
3. Else continue to the parent directory.

If both **`jaxlint.toml`** and **`pyproject.toml`** exist in the **same** directory, loading fails with **`ValueError`** (pick one file).

## Section layout by file

| File | Sections |
|------|----------|
| **`jaxlint.toml`** | Native **`[jaxlint]`** (recommended) **or** legacy **`[tool.jaxlint]`**. **Do not use both** in the same file — loading fails with a clear error. |
| **`pyproject.toml`** | **`[tool.jaxlint]`** only. A top-level **`[jaxlint]`** table is rejected — configure jaxlint under **`[tool.jaxlint]`** per PEP 621 tooling conventions. |

Equivalent settings under **`[jaxlint]`** vs **`[tool.jaxlint]`** in **`jaxlint.toml`** produce the same parsed **`LintConfig`** (including disk cache fingerprints).

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
| `cache-dir` | string (path) | *(unset)* | Optional directory for caching **`jaxlint check`** diagnostics. Relative paths are resolved against the directory containing the loaded config file (`jaxlint.toml` or `pyproject.toml`). **`jaxlint check --cache-dir`** overrides this when the CLI flag is present. |
| `severity-overrides` | table | `{}` | Map rule-id **patterns** → `error`, `warning`, or `info`. Applied **after** sensors (including **`strict-shapes`** for JL005) and **after** `select` / `ignore` / `per-file-ignores`. Pattern matching uses the same rules as `select` / `ignore` (`pattern_matches`). Keys must match at least one known rule id (unknown keys are rejected at parse time). Invalid severity strings are rejected at parse time. |
| `sarif-rule-docs-base` | string (URL) | *(unset)* | When set, **`--format sarif`** includes a **`helpUri`** on each **`tool.driver.rules`** entry: the base URL (trailing `/` optional) plus **`/`** and the **`rule_id`**. Must be **`https://`**; empty strings and non-HTTPS URLs are rejected at parse time. |
| `per-file-ignores` | table | `{}` | Map of **glob patterns** → rule patterns to disable for matching paths. |

### `select`, `ignore`, and `per-file-ignores` patterns

Matching is implemented by `pattern_matches` in `jaxlint.core.config`:

- Exact: `JL001` matches only `JL001`.
- Prefix: a pattern ending with `*` matches rule ids that **start with** the prefix before `*`. Example: `JL` matches every id starting with `JL`; `JL*` does the same.

`per-file-ignores` keys are matched with `fnmatch` against the **string form** of the file path passed to `rule_enabled` (typically relative paths like `tests/fixtures/foo.py`). Values can be a **single string** or a **list of strings**; each entry uses the same pattern rules as `ignore`.

## Examples

**Standalone config file (`jaxlint.toml`) — native table:**

```toml
[jaxlint]
select = ["JD", "JM"]
```

**Project manifest (`pyproject.toml`) — PEP 621 tool table:**

```toml
[tool.jaxlint]
select = ["JD", "JM"]
```

**Strict shape checks (JL005 as errors), native `jaxlint.toml`:**

```toml
[jaxlint]
strict-shapes = true
```

**Downgrade JL005 to a warning even when `strict-shapes` is true** (override wins):

```toml
[jaxlint]
strict-shapes = true

[jaxlint.severity-overrides]
JL005 = "warning"
```

The same layout under **`[tool.jaxlint]`** / **`[tool.jaxlint.severity-overrides]`** remains valid in **`jaxlint.toml`** for backward compatibility.

**Ignore JL005 everywhere except tune per directory:**

```toml
[jaxlint]
ignore = ["JL005"]

[jaxlint.per-file-ignores]
"experimental/**" = ["JL012"]
"kernels/**/*.py" = ["JL005", "JL010"]
```

**Custom section order for JD003:**

```toml
[jaxlint]
section-order = ["Parameters", "Returns", "Examples", "Raises"]
```
