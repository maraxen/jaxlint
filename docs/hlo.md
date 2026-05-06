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

Merge-blocking CI runs **`pytest -m needs_jax`** against this stack; some tests load **pinned StableHLO snippets** from `tests/fixtures/hlo/` (see [Contributing — HLO golden fixtures](contributing.md#hlo-golden-fixtures)).

## Goldens: mapping vs counts

`SemanticAnalyzer` exposes three overlapping views of the same MLIR text:

| View | API | Semantics |
|------|-----|-----------|
| **Operation counts** | `get_operation_inventory()` | Counts every **`stablehlo.*`** opcode substring (via regex); keys are full op names (`stablehlo.add`, …), values are **occurrence counts**. |
| **Ordered operations** | `get_operations()` | Structured **`OperationInfo`** rows in **source order** (one row per matched defining op line). |
| **Source mapping** | `get_source_mapping()` | Maps **`stablehlo.op`** name → Python-style **`loc("…")`** string when location metadata resolves; **last match wins** when the same op name appears multiple times or multiple locations compete (implementation walks regex matches and overwrites dict entries). |

Goldens may assert **inventory** hashes/text, **mapping** snippets, or **operation lists**. Those snapshots can disagree at the level of “how many times X appears” vs “which location won for op X” — that is expected: use this table when extending fixtures so expectations match the API you assert against. Feasibility tracking for deeper HLO diagnostics lives in [ADR 0001](adr/0001-hlo-diagnostics-feasibility.md).

## CLI hint

`jaxlint hlo` does not analyze a kernel by itself; it verifies the extra is importable and prints a reminder to call `GraphAuditSensor` from Python. Missing installs exit with code **`2`** (see [CLI](cli.md)).

## Separation from `jaxlint.core`

`jaxlint.core` intentionally avoids importing JAX so lightweight environments can still run JL/JD/JM checks. Reserve `jaxlint.hlo` for jobs where JAX is already a dependency.
