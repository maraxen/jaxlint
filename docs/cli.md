# CLI reference

Invoked as `jaxlint` (see `[project.scripts]` in `pyproject.toml`) or `python -m jaxlint`.

If no subcommand is given, Typer shows help (`no_args_is_help`).

## Global behavior

- **Paths:** Most commands accept zero or more file or directory paths. If omitted, the default is the current directory (`.`).
- **`--config PATH`:** Config discovery anchor (see [configuration](configuration.md)). May be a **directory** (walk upward for `jaxlint.toml` then `pyproject.toml`) or a **file** path to **`jaxlint.toml`** or **`pyproject.toml`** (parse that file only; relative paths inside the file resolve against its directory). Defaults to the current working directory when omitted.

## `jaxlint check`

Run performance (AST) sensors and, unless disabled, docstring sections + MathJax checks.

| Option | Description |
|--------|-------------|
| `PATHS...` | Files or directories (`*.py` discovered under directories). |
| `--no-doc` | Skip docstring sections and MathJax. |
| `--no-perf` | Skip AST performance sensors. |
| `--format`, `-f` | `text` (default), `compact` (one line per diagnostic: `path:line:col: severity: message (rule_id)`), `json`, `github`, **`sarif`** ([SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html), alias `sarif-2.1.0` — best-effort; not validated for GitHub Advanced Security), or **`rich`** (table layout via [Rich](https://rich.readthedocs.io/); uses plain box drawing when stdout is not a TTY). |
| `--config` | Config file (`jaxlint.toml` / `pyproject.toml`) or directory to search from. |
| `--select` | Comma-separated rule ID **prefixes**; overrides config `select`. |
| `--ignore` | Comma-separated prefixes to ignore; overrides config `ignore`. |
| `--jobs`, `-j` | Number of parallel workers over **files** (thread pool). Default: `min(32, CPU count)` (see `os.process_cpu_count()`). Use **`-j 1`** for a fully serial run (no pool). |
| `--cache-dir` | Optional directory for an **on-disk cache** of per-file diagnostics (JSON). Speeds up repeated `check` runs when sources are unchanged. Overrides **`[tool.jaxlint] cache-dir`** when passed. Invalid or corrupt cache entries are ignored (fail-open). |

**Exit code:** `1` if any diagnostic has severity `error`; otherwise `0`. Invalid `--format` exits **`2`** (same as `jaxlint hlo` when the HLO extra is missing).

## `jaxlint doc`

Docstring sections + MathJax only (no AST performance sensors).

| Option | Description |
|--------|-------------|
| `PATHS...` | As in `check`. |
| `--format`, `-f` | `text`, `compact`, `json`, `github`, **`sarif`**, or **`rich`**. Same semantics as `check`. |
| `--config` | As in `check` (directory walk or explicit config file). |
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
| `--config` | As in `check`. |

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

With `--format sarif` (alias `sarif-2.1.0`), output is a minimal [SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html) log: top-level **`$schema`**, **`version`**: `"2.1.0"`, and **`runs[0]`** with **`tool.driver`** for `jaxlint` and **`results`**. Each result includes **`ruleId`**, SARIF **`level`** (`error` / `warning` / `note` — `info` diagnostics map to `note`), **`message.text`**, and **`locations[].physicalLocation`** with **`artifactLocation.uri`** as an absolute `file:` URI and **`region`** (`startLine`, `endLine`; **`startColumn` / `endColumn`** only when the span is one line — same convention as GitHub annotations above). This is **best-effort for generic SARIF tooling** and is **not guaranteed** for GitHub Advanced Security upload requirements.

With **`--format rich`**, output is a UTF-8 text table (**Severity**, **Rule**, **Location** as `path:line:col`, **Message**). When there are no diagnostics, a single line reads **`No diagnostics.`** When standard output is a TTY and **`NO_COLOR`** is unset, severity cells use color; otherwise output is uncolored, and the table uses ASCII-safe borders when stdout is not a TTY (see `format_rich` in `jaxlint.core.runner`).
