# CLI reference

Invoked as `jaxlint` (see `[project.scripts]` in `pyproject.toml`) or `python -m jaxlint`.

If no subcommand is given, Typer shows help (`no_args_is_help`).

## Global behavior

- **Paths:** Most commands accept zero or more file or directory paths. If omitted, the default is the current directory (`.`).
- **`--config PATH`:** Directory used as the starting point when searching upward for `pyproject.toml` (see [configuration](configuration.md)). Defaults to the current working directory.

## `jaxlint check`

Run performance (AST) sensors and, unless disabled, docstring sections + MathJax checks.

| Option | Description |
|--------|-------------|
| `PATHS...` | Files or directories (`*.py` discovered under directories). |
| `--no-doc` | Skip docstring sections and MathJax. |
| `--no-perf` | Skip AST performance sensors. |
| `--format`, `-f` | `text` (default), `json`, or `github` (GitHub Checks-style JSON; see below). |
| `--config` | `pyproject.toml` search anchor (directory). |
| `--select` | Comma-separated rule ID **prefixes**; overrides config `select`. |
| `--ignore` | Comma-separated prefixes to ignore; overrides config `ignore`. |

**Exit code:** `1` if any diagnostic has severity `error`; otherwise `0`. Invalid `--format` exits **`2`** (same as `jaxlint hlo` when the HLO extra is missing).

## `jaxlint doc`

Docstring sections + MathJax only (no AST performance sensors).

| Option | Description |
|--------|-------------|
| `PATHS...` | As in `check`. |
| `--format`, `-f` | `text`, `json`, or `github`. |
| `--config` | As in `check`. |
| `--select`, `--ignore` | As in `check`. |

**Exit code:** `1` if any `error`; else `0`. Invalid `--format` exits **`2`**.

## `jaxlint score`

Rubric scoring for docstrings (see `RubricScorer` in `jaxlint.core.doc.rubric_impl`).

| Option | Description |
|--------|-------------|
| `PATHS...` | Python files to walk; default `.`. |
| `--jsonl PATH` | Score entries from a JSONL file (see implementation for expected keys). |
| `--baseline PATH` | With `--jsonl`, print mean/pass-rate comparison when the file exists. |
| `--threshold FLOAT` | Override config `rubric-threshold` (default from `[tool.jaxlint]`). |
| `--config` | `pyproject.toml` search anchor. |

**Exit code:** `1` if any scored item is below the threshold (file mode or JSONL mode); `0` otherwise. Syntax errors in a scanned `.py` print a message and count as failure for the file-walk path.

## `jaxlint rules`

Print TSV-style lines: rule id, severity, description. Always exits `0`.

## `jaxlint init`

Print a starter `[tool.jaxlint]` TOML snippet to stdout. Exits `0`.

## `jaxlint version`

Print package version. Exits `0`.

## `jaxlint hlo`

If `jaxlint[hlo]` is installed, prints a short hint for using `jaxlint.hlo` from Python. Exits `0`.

If the optional HLO stack cannot be imported (missing extra or JAX), prints an error to stderr and exits **`2`**.

## JSON output shape

For `check` and `doc` with `--format json`, output is a JSON array of objects with keys: `rule_id`, `severity`, `message`, `path`, `line`, `col`, `end_line`, `end_col` (see `format_json` in `jaxlint.core.runner`).

With `--format github`, output is a JSON object with an `annotations` array. Each element uses GitHub check run field names (`start_line`, `end_line`, `annotation_level`, `message`, `title`, `path`) suitable for tooling that posts [check run annotations](https://docs.github.com/en/rest/checks/runs). Severity maps to `annotation_level`: `error` → `failure`, `warning` → `warning`, `info` → `notice`. Single-line diagnostics may include `start_column` / `end_column` (**1-based**, per GitHub); **`start_line` ≠ `end_line`** omits columns, matching the API restriction on multiline annotations.
