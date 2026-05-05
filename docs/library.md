# Library usage (`jaxlint.core`)

Import from **`jaxlint.core`** for the stable façade (no JAX dependency):

```python
from pathlib import Path

from jaxlint.core import (
    Diagnostic,
    Severity,
    describe_rules,
    format_compact,
    format_diagnostics,
    format_github_annotations_json,
    format_json,
    format_sarif,
    format_text,
    resolve_paths,
    run_checks,
)
from jaxlint.core.config import LintConfig, load_config
```

## Load configuration

```python
cfg = load_config(Path("/your/project"))  # walks upward for pyproject.toml
# or
cfg = LintConfig.default()
cfg = LintConfig(select=frozenset(["JL"]), strict_shapes=True)
```

## Run checks

```python
paths = [Path("src/myproject")]
diags = run_checks(paths, cfg=cfg, perf=True, doc=True, mathjx=True)
# Optional: jobs=None (default parallel budget) or jobs=1 (serial), jobs>1 (thread pool).
```

- **`run_checks` return order:** The list is **pre-sorted** by `Diagnostic.sort_key()` (`path`, `line`, `column`, `rule_id`). Callers should not rely on discovery or insertion order.
- **`jobs`:** `None` uses `min(32, os.process_cpu_count() or 1)`; `1` forces serial file processing; larger values use a `ThreadPoolExecutor` over files. `LintConfig` must be treated as **read-only** across workers.
- **`perf`:** AST performance sensors (JL rules).
- **`doc`:** Docstring sections (JD rules, plus style integration).
- **`mathjx`:** MathJax diagnostics (JM rules) layered on docstring text.
- **`config_start`:** Optional anchor for `load_config` when `cfg` is omitted.

Filter by severity in Python:

```python
errors = [d for d in diags if d.severity == Severity.ERROR]
```

## Format results

```python
print(format_text(diags))
print(format_json(diags))
print(format_github_annotations_json(diags))
print(format_diagnostics(diags, "github"))  # same as format_github_annotations_json
print(format_diagnostics(diags, "sarif"))
print(format_diagnostics(diags, "compact"))
```

`format_*` helpers sort again by the same key for display, so output stays deterministic even if you build a list manually.

`format_diagnostics(diags, fmt)` accepts `text`, `compact`, `json`, `github` (alias: `github-annotations`), `sarif` (alias: `sarif-2.1.0`) and raises `ValueError` on unknown `fmt`.

## Rule catalogue

- `RULES` and `describe_rules()` expose the installed rule ids (aligned with `jaxlint rules` CLI).
- Human-oriented table: [rules.md](rules.md).

## Docstring rubric

The CLI uses `RubricScorer` from **`jaxlint.core.doc.rubric_impl`** for `jaxlint score`. Import that class directly if you need the same normalization outside the CLI.

## Optional HLO

See [hlo.md](hlo.md). Do **not** import `jaxlint.hlo` in code paths that must stay JAX-free.
