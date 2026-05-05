# HLO extra (`jaxlint[hlo]`)

StableHLO- and audit-oriented helpers live under **`jaxlint.hlo`**. Installing them pulls in **`jax`** via the optional dependency group **`hlo`**.

```bash
uv sync --extra hlo
# or
pip install "jaxlint[hlo]"
```

## Imports

From `jaxlint.hlo`:

- `GraphAuditSensor` — kernel graph audit hooks (see `jaxlint/hlo/graph_audit.py`).
- `SemanticAnalyzer`, `StableHLOGraph` — parsing/summarization utilities (see `jaxlint/hlo/stablehlo_analysis.py`).

```python
from jaxlint.hlo import GraphAuditSensor, SemanticAnalyzer, StableHLOGraph
```

Actual behavior depends on JAX version, lowered IR, and how you invoke `audit_kernel`; treat this module as **runtime-sensitive**.

## CLI hint

`jaxlint hlo` does not analyze a kernel by itself; it verifies the extra is importable and prints a reminder to call `GraphAuditSensor` from Python. Missing installs exit with code **`2`** (see [CLI](cli.md)).

## Separation from `jaxlint.core`

`jaxlint.core` intentionally avoids importing JAX so lightweight environments can still run JL/JD/JM checks. Reserve `jaxlint.hlo` for jobs where JAX is already a dependency.
