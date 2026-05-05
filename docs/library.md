# Library usage (`jaxlint.core`)

Import from **`jaxlint.core`** for the stable façade (no JAX dependency):

```python
from pathlib import Path

from jaxlint.core import (
    Diagnostic,
    Severity,
    describe_rules,
    format_json,
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
```

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
```

Diagnostics sort by `(path, line, column, rule_id)` for deterministic output.

## Rule catalogue

- `RULES` and `describe_rules()` expose the installed rule ids (aligned with `jaxlint rules` CLI).
- Human-oriented table: [rules.md](rules.md).

## Docstring rubric

The CLI uses `RubricScorer` from **`jaxlint.core.doc.rubric_impl`** for `jaxlint score`. Import that class directly if you need the same normalization outside the CLI.

## Optional HLO

See [hlo.md](hlo.md). Do **not** import `jaxlint.hlo` in code paths that must stay JAX-free.
