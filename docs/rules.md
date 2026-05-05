# jaxlint rule catalogue

| ID | Severity | Description |
|----|-----------|-------------|
| JL001 | error | Scatter+gather: `.at[].set()` then immediate subscript of the result. |
| JL002 | warning | Short function with `@jax.jit` (helper fusion risk). |
| JL003 | error | `for`/`while` inside `@jax.jit` / `@jax.grad`. |
| JL004 | error | `print()` / `open()` inside a JAX kernel. |
| JL005 | warning | `.shape` / `jnp.shape` in kernel (error with `strict-shapes`). |
| JL010 | warning | `jnp.float64` usage. |
| JL011 | warning | `ShardingSpec` usage. |
| JL012 | warning | `/` in JIT kernel. |
| JD001 | error | Missing docstring on public function/class. |
| JD002 | error | Missing required doc section (from config). |
| JD003 | warning | Section order vs `section-order` in config. |
| JD004 | warning | Parameters in code vs Parameters block mismatch. |
| JD005 | warning | Return annotation without Returns section. |
| JM001 | error | Unbalanced `$` / `$$` math delimiters. |
| JM002 | error | Malformed LaTeX (braces, environments, `\\frac`). |
| JM003 | warning | Large `$$` block with little prose. |

Configure rules via `[tool.jaxlint]` in `pyproject.toml`.
