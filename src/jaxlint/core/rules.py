"""Stable rule catalogue for `jaxlint rules` and documentation."""

from __future__ import annotations

from jaxlint.core.diagnostic import Severity

# (rule_id, severity, short description)
RULES: list[tuple[str, Severity, str]] = [
    ("JL001", Severity.ERROR, "Scatter+gather anti-pattern: `.at[].set()` result immediately subscripted."),
    ("JL002", Severity.WARNING, "Short function has `@jax.jit` — may block fusion if used as helper."),
    ("JL003", Severity.ERROR, "Python `for`/`while` inside a `@jax.jit` or `@jax.grad` function."),
    ("JL004", Severity.ERROR, "Side effect (`print`/`open`) inside a JAX kernel."),
    (
        "JL005",
        Severity.WARNING,
        "`.shape` / `jnp.shape` inside a JAX kernel — can break shape polymorphism (use --strict-shapes for error).",
    ),
    ("JL010", Severity.WARNING, "`jnp.float64` — prefer f32/bf16 unless double precision is required."),
    ("JL011", Severity.WARNING, "`ShardingSpec` — prefer NamedSharding + Mesh."),
    ("JL012", Severity.WARNING, "Division `/` in JIT kernel — consider safe reciprocal if divisor may be zero."),
    ("JD001", Severity.ERROR, "Public function or class has no docstring."),
    ("JD002", Severity.ERROR, "Docstring missing required section."),
    ("JD003", Severity.WARNING, "Section order does not match canonical order for style."),
    ("JD004", Severity.WARNING, "Docstring Parameters do not cover all function arguments."),
    ("JD005", Severity.WARNING, "Function has return annotation but docstring lacks Returns section."),
    ("JM001", Severity.ERROR, "Math delimiter imbalance or invalid $/$$ pairing (outside code fences)."),
    ("JM002", Severity.ERROR, "Malformed LaTeX structure in MathJax region."),
    ("JM003", Severity.WARNING, "Large math block with little surrounding prose."),
]

KNOWN_RULE_IDS: frozenset[str] = frozenset(rule_id for rule_id, _, _ in RULES)


def describe_rules() -> str:
    lines = []
    for rid, sev, desc in RULES:
        lines.append(f"{rid}\t[{sev.value}]\t{desc}")
    return "\n".join(lines)
