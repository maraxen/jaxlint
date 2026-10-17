# jaxlint rule catalogue

Configure rules via **`[jaxlint]`** / **`[tool.jaxlint]`** (see [configuration.md](configuration.md)). Use `jaxlint rules` for the authoritative list bundled with your install.

## Summary table

| ID | Severity | When it fires | Example fixture |
|----|----------|----------------|----------------|
| JL001 | error | `.at[].set()` result immediately subscripted | `tests/fixtures/jl001_scatter_gather.py` |
| JL002 | warning | `@jax.jit` on a very small helper body | `tests/fixtures/jl002_short_jit.py` |
| JL003 | error | Python `for`/`while` inside `@jax.jit` / `@jax.grad` | `tests/fixtures/jl003_loop_kernel.py` |
| JL004 | error | `print`/`open`-style side effects in a kernel | `tests/fixtures/jl004_side_effect.py` |
| JL005 | warning / error* | `.shape` / `jnp.shape` inside kernel; *errors if `strict-shapes` | `tests/fixtures/jl005_shape_in_kernel.py` |
| JL010 | warning | `jnp.float64` usage | `tests/fixtures/jl010_float64.py` |
| JL011 | warning | `ShardingSpec` usage | `tests/fixtures/jl011_sharding_spec.py` |
| JL012 | warning | Bare `/` in JIT kernel | `tests/fixtures/jl012_div_jit.py` |
| JD001 | error | Public function/class missing docstring | `tests/fixtures/jd001_missing_docstring.py` |
| JD002 | error | Missing required doc section (`required-sections`) | `tests/fixtures/jd002_missing_parameters.py` |
| JD003 | warning | Section order disagrees with `section-order` | `tests/fixtures/jd003_section_order.py` |
| JD004 | warning | Parameters in signature vs Parameters block mismatch | `tests/fixtures/jd004_param_mismatch.py` |
| JD005 | warning | Return annotation without Returns section | `tests/fixtures/jd005_no_returns_section.py` |
| JM001 | error | Unbalanced `$`/`$$` (outside fenced code) | `tests/fixtures/jm001_bad_math.py` |
| JM002 | error | Malformed LaTeX/MathJax structure | `tests/fixtures/jm002_bad_latex.py` |
| JM003 | warning | Large `$$` math block with little prose | `tests/fixtures/jm003_large_math.py` |

## Suppressing a diagnostic inline

Add a comment on the line a diagnostic is reported on:

```python
for i in range(3):  # jaxlint: ignore[JL003]
    x = x + i
y = x / 2  # jaxlint: ignore[JL012]
z = x / 3  # jaxlint: ignore            (bare: all rules on this line)
```

- Rule IDs are case-sensitive; whitespace inside the brackets is ignored.
- Only the reported line is affected. A comment on the `def` line does not silence a loop on a later line.
- Unknown IDs are ignored silently.
- Comments are found with `tokenize`, so `# jaxlint: ignore` inside a string literal has no effect.
- The filter runs for the CLI, the LSP, the MCP server and `run_checks`, and the disk cache is keyed on file content, so an added comment takes effect immediately.

## Detailed descriptions

| ID | Description |
|----|-------------|
| JL001 | Scatter+gather anti-pattern: `.at[].set()` then immediate slice/subscript on the result. |
| JL002 | Short function has `@jax.jit` — may block fusion if used repeatedly as an inner helper. |
| JL003 | Python `for`/`while` traced inside JAX transforms without `scan`/structured loops. |
| JL004 | Side effects (`print`, `open`, …) referenced inside kernels. |
| JL005 | Shape introspection (`x.shape`, `jnp.shape`) breaks polymorphism unless carefully justified. |
| JL010 | `float64` doubles memory and slows many GPU kernels versus `float32`/`bfloat16`. |
| JL011 | `ShardingSpec` is legacy; prefer `NamedSharding` with an explicit mesh. |
| JL012 | Division may introduce NaNs; consider guarded reciprocals where denominators vanish. |
| JD001 | Public API should document intent; private names (`_*`) ignored except dunders. |
| JD002 | Enforces `required-sections`, including special cases like Pseudocode/Hardware Notes. |
| JD003 | Parsed section kinds appear out of the configured canonical order (`section-order`). |
| JD004 | Doc parameters must mirror the callable signature (minus `self`/`cls`). |
| JD005 | Annotated returns should document the returned value/object. |
| JM001 | Stray `$`/`$$` delimiters confuse MathJax renderers and readers. |
| JM002 | Structural LaTeX issues (balanced braces/environments/macros such as `\frac`). |
| JM003 | Very large display math compared to explanatory prose hurts readability/scannability. |
