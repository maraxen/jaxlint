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
    ("JL022", Severity.WARNING, "jit-family wrap of a fresh callable inside a loop (retraces every iteration)."),
    ("JD001", Severity.ERROR, "Public function or class has no docstring."),
    ("JD002", Severity.ERROR, "Docstring missing required section."),
    ("JD003", Severity.WARNING, "Section order does not match canonical order for style."),
    ("JD004", Severity.WARNING, "Docstring Parameters do not cover all function arguments."),
    ("JD005", Severity.WARNING, "Function has return annotation but docstring lacks Returns section."),
    ("JM001", Severity.ERROR, "Math delimiter imbalance or invalid $/$$ pairing (outside code fences)."),
    ("JM002", Severity.ERROR, "Malformed LaTeX structure in MathJax region."),
    ("JL021", Severity.INFO, "Host loop over replicates/resamples — batch the replicate axis."),
    ("JM003", Severity.WARNING, "Large math block with little surrounding prose."),
]

# rule_id -> what to do after the rule fires. Host-side rules name a measurement step:
# a static suspect is confirmed or cleared by `xtrax profile` (spec 261016, Workstream 1).
# Emitted as SARIF ``help`` text; rules without an entry carry no followup.
RULE_FOLLOWUPS: dict[str, str] = {
    "JL022": "Confirm with `xtrax profile`: steady-state n_compilations > 0 per unit is the retrace signature. Fix by building the jitted function once outside the loop and passing loop-varying values as arguments.",
    "JL021": "Confirm with `xtrax profile <entrypoint> --scale-axis <replicate axis>`: exec/element >= 0.5 is the host-loop signature. Fix by batching the replicate axis (jax.vmap, or batched NumPy/einsum on host) and test batched == looped output.",
}

KNOWN_RULE_IDS: frozenset[str] = frozenset(rule_id for rule_id, _, _ in RULES)


def describe_rules() -> str:
    lines = []
    for rid, sev, desc in RULES:
        lines.append(f"{rid}\t[{sev.value}]\t{desc}")
    return "\n".join(lines)
