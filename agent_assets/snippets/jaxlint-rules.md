# JAX Linting Rules

Use jaxlint to catch common JAX anti-patterns early. Key rules to remember:

## Tracing Constraints (JL-series, often ERROR)

**JL003** (Python control flow in kernel): Never use `if`, `for`, or `while` inside `@jax.jit` or `@jax.grad` on tracer values. Use `jax.lax.cond()`, `jax.lax.fori_loop()`, or `jax.lax.scan()` instead. Move host-side loops outside the kernel.

**JL004** (I/O inside kernel): No `print()`, `open()`, or file operations inside JAX kernels. Move all I/O outside the JIT boundary. Use `jax.debug.print()` for runtime inspection inside JIT.

**JL005** (Shape inside kernel): Accessing `.shape` or `jnp.shape()` inside a kernel breaks shape polymorphism. Pass shapes as arguments instead. If shape-dependent logic is needed, use `jax.lax.cond()` with static bounds.

**JL001** (Scatter-gather anti-pattern): Don't immediately subscript the result of `.at[].set()` — this becomes a scatter followed by a gather, wasting memory. Restructure to keep arrays in JAX until final output.

## Equinox Module Patterns

Use `eqx.filter_jit` and `eqx.filter_grad` on `eqx.Module` instances instead of bare `jax.jit`/`jax.grad`. Filter transforms treat non-array fields (marked `eqx.field(static=True)`) as static, avoiding unnecessary recompilation:

```python
# Right: filter_jit respects static fields
@eqx.filter_jit
def forward(model: Linear, x):
    return model(x)

# Wrong: jax.jit tries to trace through weight dtype (static)
@jax.jit
def forward(model: Linear, x):
    return model(x)
```

Mark config fields (layer sizes, activation names, boolean flags) with `eqx.field(static=True)` to prevent shape/dtype changes from forcing recompilation.

## NaN/Inf Debugging

Never use `print()` inside `@jax.jit` — it only executes once at trace time. Use `jax.debug.print()` instead:

```python
@jax.jit
def compute(x, y):
    jax.debug.print("x shape: {x}", x=x)  # Runs at execution time
    result = x / y
    jax.debug.print("result: {result}", result=result)  # Shows NaN if it appears
    return result
```

Add epsilon guards before division and use `jnp.where()` to mask invalid values. Check for divide-by-zero before reciprocal operations.

## When to Run jaxlint

Run `jaxlint check src/` before submitting to CI, code review, or cluster jobs. Lint early and often — catching anti-patterns before merge prevents performance regressions and subtle debugging sessions.

Integration: Use `--select JL src/` to check only performance rules (skip doc checks in dev). Use `--format json` for CI/automation pipelines.

## Shape Annotation Convention

Keep shapes explicit in function signatures when they vary per use case. Document batch dimensions, feature dimensions, and sequence lengths in docstrings to prevent shape mismatches.
