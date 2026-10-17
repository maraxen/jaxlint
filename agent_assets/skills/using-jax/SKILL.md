---
name: using-jax
description: Use when writing or modifying JAX code — tracer errors, NaN debugging, recompilation issues, memory pressure, or jaxlint rule violations — and before writing any loop over replicates, bootstrap resamples, pairs, edges, cells or frames, or launching a numerical run expected to take more than a few minutes (vectorization + production-shape profiling discipline).
when_to_use: 'Triggers: tracer error, jit, vmap, NaN, recompile.'
---
# using-jax

## When to Use This Skill

Activate this skill when writing JAX code at the function/kernel level (single process, one or a few devices), or when jaxlint surfaces rule violations (JL001, JL002, etc.). Triggers include: tracer errors, NaN/Inf propagation, out-of-memory during JIT, unexpected recompilation, shape mismatches, or performance problems. **Do not use for**: multi-host distributed (see future distributed-jax skill), IREE export (see exporting-jax), or Equinox modules (see using-equinox).

## Symptom-to-Fix Table (Quick Reference)

Refer to this table first when debugging JAX errors:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ConcretizationTypeError: XXX weakly-typed` | Tracer used in Python control flow | Replace `if x > 0:` with `jax.lax.cond(x > 0, ...)` |
| `TracerBoolConversionError` | Using Python `if` / `while` on tracer | Use `jax.lax.cond`, `jax.lax.while_loop`, or mark arg `static_argnums` |
| `TypeError: only integers and slices` | Tracer used as array index | Use `jax.lax.dynamic_slice`, scatter, or gather operations |
| `LeakedTraceError` | Tracer escaped to mutable state (list, global, etc.) | Don't store tracers in lists, dicts, or globals; return only arrays |
| Unexpected recompilation | Shape/dtype changes or Python object identity drift | Mark Python args with `static_argnames`, ensure dtypes consistent |
| NaN appears mid-computation | Division by zero, log of negative, masked invalid values | Use epsilon guards, `jnp.where` masking, check for zero before reciprocal |
| Out of memory during JIT | Intermediate activations materialized | Wrap layers with `jax.checkpoint()` or use `jax.lax.scan` with carry |
| `XlaRuntimeError: RESOURCE_EXHAUSTED` | Execution OOM (insufficient device memory) | Reduce batch size, use `donate_argnums` for buffer reuse, or apply checkpointing |
| Opaque/cryptic XLA compile error | JAX traceback filtering obscures root cause | Set `JAX_TRACEBACK_FILTERING=off` environment variable and re-run |
| "Function called with different static argument values" | Static arg cache thrashing (too many unique values) | Stabilize static args (e.g., normalize enum values, avoid float keys) |
| Mixing `np.ndarray` and `jnp.ndarray` types | Host transfer and type mismatch | Convert all inputs to `jnp` upfront: `jnp.asarray(...)` at function entry |
| `tracer was never converted to array` | Printing or using tracer in Python context | Use `jax.debug.print()` instead of `print()`, or extract value with `x.item()` (loses compilation) |

## JIT Mental Model

JAX's `jax.jit` splits code into two phases: **trace time** and **execution time**. During trace, JAX records a symbolic computation graph using abstract "tracers" that represent shapes and dtypes, not values. Code that prints tracer objects or branches on their values hits concrete ization errors.

**Tracing vs execution:**
- Print statements inside `@jax.jit` execute at trace time once, then disappear in compiled code—use `jax.debug.print()` instead.
- Python control flow (`if`, `while`) cannot depend on tracer values; use `jax.lax.cond` / `jax.lax.while_loop` for dynamic branching.
- Recompilation happens whenever shape, dtype, or Python object identity changes; minimize by using `static_argnums` and `static_argnames`.

**Donation example:**
```python
import jax
import jax.numpy as jnp

@jax.jit
def log_and_add(x, y):
    print(x)  # WRONG: prints tracer at trace time, then nothing at execution
    return x + y

result = log_and_add(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
print(result)  # Only prints result

# Correct: use jax.debug.print for runtime inspection
@jax.jit
def log_and_add_correct(x, y):
    jax.debug.print("x = {x}", x=x)  # Prints at execution time
    return x + y

result = log_and_add_correct(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
```

**Static arguments:**
```python
import jax
import jax.numpy as jnp
from jax import random

@jax.jit
def process_batch(data, mode='train'):  # mode is Python string, not JAX array
    if mode == 'train':
        # WRONG: jnp.random.normal doesn't exist; use jax.random.normal with key
        return data
    return data

# Will recompile if you call with mode='eval' because mode is not marked static
# Fix: use static_argnames in decorator:
@jax.jit(static_argnames=['mode'])
def process_batch_static(data, mode='train', key=None):
    if mode == 'train':
        key, subkey = random.split(key)
        # Correct: explicit key argument
        return data + random.normal(subkey, data.shape)
    return data

# Now mode='train' and mode='eval' use the same compilation
# Usage:
key = random.key(0)
result = process_batch_static(jnp.ones((3, 4)), mode='train', key=key)
```

## PyTrees

A PyTree is a container (dict, list, tuple, or custom class) holding JAX arrays and other PyTrees. JAX flattens PyTrees into a list of leaves for tracing and map operations. Use `jax.tree_util.tree_map()` to apply a function to all leaves.

**Key functions:**
- `jax.tree_util.tree_map(f, tree)` — apply function to all leaves
- `jax.tree_util.tree_leaves(tree)` — extract all leaves
- `jax.tree_util.tree_structure(tree)` — get shape without values

**Example:**
```python
import jax
import jax.numpy as jnp
from jax import tree_util

# Model as nested dict (PyTree)
model = {
    'encoder': {'w': jnp.ones((3, 4)), 'b': jnp.zeros(4)},
    'decoder': {'w': jnp.ones((4, 2)), 'b': jnp.zeros(2)}
}

# Tree map: apply relu to all arrays
def zero_weights(x):
    return jnp.zeros_like(x)

zeroed = tree_util.tree_map(zero_weights, model)
print(zeroed['encoder']['w'])  # All zeros

# Tree leaves: flatten structure
leaves = tree_util.tree_leaves(model)
print(len(leaves))  # 4 (two weight matrices, two bias vectors)

# Common: scale all parameters
model_scaled = tree_util.tree_map(lambda x: x * 0.1, model)
```

## PRNG Discipline

JAX has no global random state; generate keys explicitly with `jax.random.key()` and split them for independent streams. Reusing a key produces identical sequences—always split before use. Use `jax.random.fold_in()` for parallel seed generation.

**Key rules:**
- Call `jax.random.key(seed)` once at program start.
- Split before using: `key, subkey = jax.random.split(key)`.
- Never reuse a key in the same computation.
- For batch parallelism, use `jax.vmap` with key splitting.

**Example:**
```python
import jax
import jax.numpy as jnp
from jax import random

# WRONG: reusing the same key
key = random.key(0)
x = random.normal(key, (10,))
y = random.normal(key, (10,))  # WRONG: y == x (same sequence)

# Correct: split before each use
key = random.key(0)
key, subkey1 = random.split(key)
x = random.normal(subkey1, (10,))
key, subkey2 = random.split(key)
y = random.normal(subkey2, (10,))  # Different from x

# Batched sampling with vmap: use fold_in for parallel keys
def sample_once(key_i, batch_size):
    return random.normal(key_i, (batch_size,))

key = random.key(42)
batch_keys = jax.vmap(lambda i: random.fold_in(key, i))(jnp.arange(4))
samples = jax.vmap(sample_once, in_axes=(0, None))(batch_keys, 10)
print(samples.shape)  # (4, 10)
```

## Control Flow Primitives

Python's `if` and `while` cannot depend on tracer values. Use `jax.lax.cond` for branches, `jax.lax.scan` / `jax.lax.fori_loop` for loops, and `jax.lax.while_loop` for dynamic termination. `jax.vmap` applies a function over a batch axis automatically.

**Quick reference:**
- `jax.lax.cond(pred, f_true, f_false, *operands)` — if/else for dynamic predicates
- `jax.lax.scan(f, init, xs)` — loop with carry (sequence → accumulator); reverse-mode differentiable
- `jax.lax.fori_loop(start, stop, body_fn, init)` — for loop with fixed range
- `jax.lax.while_loop(cond_fn, body_fn, init)` — while loop with dynamic termination; not reverse-mode differentiable
- `jax.vmap(f, in_axes=...)` — vectorize over batch axis

**Example:**
```python
import jax
import jax.numpy as jnp
from jax import lax

# WRONG: Python if with tracer
@jax.jit
def compute_wrong(x):
    if x > 0:  # WRONG: x is a tracer
        return x ** 2
    return -x

# Correct: use jax.lax.cond with operand-passing
@jax.jit
def compute_correct(x):
    # Pass x as operand, lambdas receive it
    return lax.cond(x > 0, lambda x: x ** 2, lambda x: -x, x)

# Simpler alternative (when both branches are cheap):
@jax.jit
def compute_alternative(x):
    return jnp.where(x > 0, x ** 2, -x)

# Loop with scan (sequential dependency)
def step(carry, x):
    state = carry + x
    return state, state

key = jnp.array([1.0, 2.0, 3.0])
final_state, trajectory = lax.scan(step, 0.0, key)
print(trajectory)  # [1.0, 3.0, 6.0]

# Batch with vmap
def predict(x, w):
    return jnp.dot(x, w)

batch_x = jnp.ones((5, 10))
w = jnp.ones(10)
predictions = jax.vmap(lambda x: predict(x, w))(batch_x)
print(predictions.shape)  # (5,)
```

## lax.scan vs lax.while_loop for Autoregressive Loops

**Choose by semantics, not by an assumed compile-time gap.**

- **IR size is the same at `unroll=1`.** Both primitives lower to a single `stablehlo.while`
  with one body (verified on jax 0.11.1: 1 `while` op each, 39 vs 29 lines of StableHLO for a
  512-step loop). `scan` is not unrolled per step. Only `unroll=k > 1` replicates the body,
  and it is still one `while`.
- **`lax.scan`** is reverse-mode differentiable and stacks per-step outputs for you. Use it
  for training paths and any fixed-length loop inside `jax.grad` / `eqx.filter_grad`.
- **`lax.while_loop`** supports data-dependent termination (stop at EOS), but it is **not
  reverse-mode differentiable**. Use it for inference paths that can stop early.
- **Long autoregressive loops on GPU: expect `scan` to be able to cost far more backend
  compile time.** Two reports:
  - prxteinmpnn, Blackwell SM120, bucket 512, B=16: 30–40 min cold compile for `scan`
    (`unroll=1`) against 6.5 s for `while_loop`;
  - xtrax, L40S, a 512-step decode: over 20 min of backend codegen as `scan`, with no such
    cost as one `WhileOp`.

  The StableHLO is the same size either way (above), so the gap arises later, in XLA's
  backend. A plausible but unverified cause: a `scan` has a static trip count XLA can unroll
  on, and a data-dependent `while_loop` does not. For an inference-only long loop, prefer
  `while_loop`. Before switching for compile time alone, measure compile seconds for both
  (`jax_log_compiles`, or `xtrax.profiling.count_backend_compiles`).

**Pattern: bundle the carry as a single struct, with a preallocated output buffer:**

```python
import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple

class ARCarry(NamedTuple):
    step: jax.Array            # scalar int, current position
    tokens: jax.Array          # (seq_len,) output buffer
    state: jax.Array           # recurrent state / KV cache

def ar_step(carry: ARCarry) -> ARCarry:
    step, tokens, state = carry
    new_token, new_state = model_step(tokens[step - 1], state)
    tokens = tokens.at[step].set(new_token)
    return ARCarry(step + 1, tokens, new_state)

def ar_decode_inference(init_tokens, init_state, max_len: int) -> jax.Array:
    init_carry = ARCarry(
        step=jnp.array(1),
        tokens=init_tokens,
        state=init_state,
    )
    final_carry = lax.while_loop(
        cond_fun=lambda c: c.step < max_len,
        body_fun=ar_step,
        init_val=init_carry,
    )
    return final_carry.tokens

# Training path — use lax.scan (differentiable). The carry keeps a fixed shape:
# carry the last token and let scan stack the per-step outputs. Never grow an
# array in the carry (jnp.append / concatenate): scan requires the carry's
# shape and dtype to be identical on every step and raises a TypeError otherwise.
def ar_decode_train(init_token, init_state, max_len: int) -> jax.Array:
    def step(carry, _):
        token, state = carry
        new_token, new_state = model_step(token, state)
        return (new_token, new_state), new_token
    _, tokens = lax.scan(step, (init_token, init_state), None, length=max_len)
    return tokens  # (max_len, ...) stacked by scan
```

**Decision table:**

| Context | Use |
|---------|-----|
| Needs gradients (training, inside `jax.grad` / `eqx.filter_grad`) | `lax.scan` |
| Inference with data-dependent stop (EOS) | `lax.while_loop` |
| Fixed-length inference | Either: same IR at `unroll=1`; measure if compile time matters |
| Both modes needed | Separate code paths keyed by a mode flag. With xtrax: `CarrySpec(collect_outputs=True)` plans a `Scan`, and `CarrySpec(collect_outputs=False, cond=...)` plans a `WhileCarry` (`using-xtrax`, `references/adoption.md`) |

## Numerical Safety

Avoid NaN gradients and numerical instability: mask invalid positions using `jnp.where` to skip NaN-producing terms, add epsilon guards to reciprocals, and enable float64 for physics code with `jax.config.update()`.

**Patterns:**
- `jnp.where(mask, safe_value, jnp.ones_like(x))` — prevent NaN in masked branches
- `1.0 / (x + 1e-8)` — epsilon guard for division
- `jax.config.update("jax_enable_x64", True)` — enable float64 (slower but higher precision)
- `jnp.sqrt(jnp.maximum(x, 0.0))` — prevent negative sqrt

**Example:**
```python
import jax
import jax.numpy as jnp

# WRONG: Zeroing logits before softmax causes NaN gradients
def softmax_masked_wrong(logits, mask):
    logits_masked = jnp.where(mask, logits, -1e9)
    return jnp.exp(logits_masked) / jnp.sum(jnp.exp(logits_masked))
# Problem: gradient NaN where mask=False (softmax undefined for all -inf)

# Correct: numerically stable masked softmax (double-where pattern)
def softmax_masked_stable(logits, mask):
    # Step 1: exclude padded positions before softmax
    masked_logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    # Step 2: zero out probs at masked positions (safe after softmax)
    probs = jnp.where(mask, probs, 0.0)
    return probs

# Handle all-masked rows (avoid NaN when all positions masked)
def softmax_masked_safe(logits, mask):
    masked_logits = jnp.where(mask, logits, -jnp.inf)
    # Add tiny epsilon to rows that would be all-masked
    safe_mask = jnp.any(mask, axis=-1, keepdims=True)
    masked_logits = jnp.where(safe_mask, masked_logits, 0.0)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    probs = jnp.where(mask, probs, 0.0)
    return probs

# Epsilon guard for division
def safe_inverse(x):
    return 1.0 / (jnp.abs(x) + 1e-10)

# Float64 for physics
jax.config.update("jax_enable_x64", True)
x = jnp.array([1.0, 2.0, 3.0])  # Now uses float64 instead of float32
```

## Memory and Checkpointing

Use `jax.checkpoint()` (a.k.a. `jax.remat`) to trade compute for memory: recompute intermediate activations instead of storing them. For transformer-like patterns, use checkpoint policies to checkpoint at appropriate granularity. Use `jax.lax.scan` with carry to avoid materializing large intermediate arrays.

**Key patterns:**
- `jax.checkpoint(f)` — recompute activations in backward pass
- `jax.lax.scan()` — loop with carry (no intermediate array materialization)
- `jax.checkpoint_policies.checkpoint_dots` — checkpoint at matmul boundaries (from `jax.ad_checkpoint`)

**Buffer donation example:**
```python
import jax
import jax.numpy as jnp
from jax import lax
import functools as ft

# Donate buffer: state buffer reused, speeds up SGD
@ft.partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch):
    # state buffer is donated to XLA — reused for output
    # DO NOT use state after this call; it may be overwritten
    new_state = state.apply_updates(grad_updates)
    return new_state

# Without checkpoint: O(depth) memory
def deep_network(x):
    for i in range(100):
        x = jnp.sin(x) + 0.1
    return x

# With checkpoint: O(1) memory, recompute in backward
def deep_network_checkpointed(x):
    def layer(x, i):
        return jnp.sin(x) + 0.1, None
    
    x, _ = lax.scan(jax.checkpoint(layer), x, jnp.arange(100))
    return x

# Scan to avoid materializing sequence
def process_sequence(carry, x):
    new_carry = carry + jnp.sum(x)
    return new_carry, None  # Don't return intermediate arrays
    
carry = lax.scan(process_sequence, 0.0, sequence)[0]
```

## Gradients and Automatic Differentiation

Use `jax.grad()` to compute gradients, `jax.value_and_grad()` to return both value and gradient, and `has_aux=True` to return auxiliary data (non-differentiable values). For masked operations, use the double-where trick to prevent NaN gradients.

**Key functions:**
- `jax.grad(f)` — compute gradient of f at a point
- `jax.value_and_grad(f, has_aux=True)` — return (value, aux), grad
- `jax.grad(f, argnums=(...))` — differentiate w.r.t. multiple arguments
- Double-where for NaN-safe masked gradients

**Example:**
```python
import jax
import jax.numpy as jnp

# Simple gradient
def loss(x):
    return jnp.sum(x ** 2)

grad_fn = jax.grad(loss)
x = jnp.array([1.0, 2.0, 3.0])
grads = grad_fn(x)
print(grads)  # [2.0, 4.0, 6.0]

# Value and gradient together
def loss_with_metrics(x):
    loss_val = jnp.sum(x ** 2)
    return loss_val, {'norm': jnp.linalg.norm(x)}

loss_fn = jax.value_and_grad(loss_with_metrics, has_aux=True)
(loss_val, aux), grads = loss_fn(x)
print(loss_val, aux, grads)

# Gradient w.r.t. multiple arguments
def loss_multi(x, w):
    return jnp.sum((x @ w) ** 2)

grad_x_w = jax.grad(loss_multi, argnums=(0, 1))
grads_x, grads_w = grad_x_w(x, w)

# NaN-safe gradient through masked positions (double-where trick)
def masked_loss(x, mask):
    # Replace masked positions with safe values (e.g., 1.0)
    safe_x = jnp.where(mask, x, jnp.ones_like(x))
    # Compute loss with safe values
    loss_val = jnp.sum((safe_x ** 2) * mask)
    # Zero gradient at masked positions (jnp.where is differentiable)
    return loss_val

grad_masked = jax.grad(masked_loss)
grads = grad_masked(x, mask)
# Gradients are zero where mask=False, non-zero where mask=True
```

## Debugging Techniques

Enable `jax_debug_nans` to print NaN locations, use `jax.debug.print()` for runtime inspection inside JIT, and `jax.debug.breakpoint()` for interactive debugging. Use `jax.block_until_ready()` for correct benchmarking. Compile with AOT to profile and inspect the computation graph.

**Commands:**
- `jax.config.update("jax_debug_nans", True)` — halt on NaN
- `jax.debug.print("{x}", x=x)` — print inside JIT
- `jax.debug.breakpoint()` — enter pdb inside JIT
- `jax.block_until_ready(result)` — force execution (needed for correct timing)
- `jax.jit(f).lower(x).as_text()` — inspect StableHLO IR

**Example:**
```python
import jax
import jax.numpy as jnp
import time

# Enable NaN debugging
jax.config.update("jax_debug_nans", True)

@jax.jit
def compute_with_nan(x):
    y = 1.0 / (x - 1.0)  # NaN when x=1.0
    jax.debug.print("y = {y}", y=y)
    return y

# This will halt and show NaN location
compute_with_nan(jnp.array(1.0))

# Correct benchmarking with block_until_ready
f = jax.jit(lambda x: x ** 2)
x = jnp.array([1.0, 2.0, 3.0])

# WRONG: measures compilation time, not execution
start = time.time()
result = f(x)
elapsed_wrong = time.time() - start

# Correct: force execution before timing
start = time.time()
result = f(x)
jax.block_until_ready(result)
elapsed_correct = time.time() - start

# StableHLO inspection
lowered = jax.jit(lambda x: x ** 2).lower(x)
print(lowered.as_text())  # View IR
```


## jaxlint Feedback Loop

Run `jaxlint check src/` to surface JAX anti-patterns and rule violations. **JL001** flags scatter/gather patterns that should use `jnp.where` or structured indexing; **JL002** flags missing JIT decorators on hot functions. Address violations by rewriting patterns (e.g., replace scatter loops with vectorized operations) or adding explicit JIT. For the full rule catalog and detailed guidance on each rule, see the `using-jaxlint` skill.

**Quick example:**
```bash
jaxlint check src/
# JL001: Scatter/gather detected in src/model.py:45 — rewrite with jnp.where
# JL002: Hot loop missing @jax.jit in src/utils.py:12
```

Interpret output as actionable refactoring: rewrite scatter patterns to vectorized form, mark hot functions with JIT, remove Python control flow that depends on tracers.

## Vectorize everything you can — and profile at production shapes

*(Moved verbatim from the `~/.claude/CLAUDE.md` hard rule, 261016. The always-on core lives in
`~/.claude/rules/vectorize.md`.)*

**A Python loop over replicates, bootstrap resamples, pairs, edges, cells or frames is a defect
until shown otherwise.** Written into the first design, not retrofitted after a slow run.

- **Batch the leading axis.** Express B bootstrap draws, R replicates, P pairs and G grid
  points as one array axis (`jax.vmap`, `lax.scan` only for true recurrences, batched
  NumPy/`bincount`/einsum on host), not `for b in range(B): f(x_b)`. Where a function is
  written for one input, give it a batched form and test that the batched output equals the
  looped output bit for bit (or to a stated tolerance) before trusting it.
- **Profile before running anything that takes more than a few minutes.** A tiny tracked
  probe at production shapes (production K, N, B and pair count, not toy sizes) that reports
  seconds per unit and the split between compile, device work, host work and Python
  overhead. Estimate total wall time from it and put the estimate in the sidecar. A probe at
  toy shapes misleads: cost can scale with K or K² and hide there.
- **Count recompiles.** Set `jax_log_compiles` in the probe and confirm a steady-state unit
  triggers zero. Keep shapes static (pad/mask, do not vary shapes per unit), hoist `jit`
  out of loops, and do not spawn a fresh process per tiny unit if its start-up and compile
  cost is comparable to the unit.
- **Host loops over device calls are the usual culprit.** A per-unit device round trip or a
  host-side numerical routine called B×P×arms times is the first thing to batch or move
  on-device. Keep float64 on host for small post-hoc sums only; no device x64 in hot loops.
- **Do not parallelise a slow algorithm in place of fixing it.** More workers on a shared
  box (titanix) only raise load. Vectorise first, then size workers to the cores that are
  actually free.

(Learned 2026-10-16, denxity QB-MI coverage run: each bootstrap replicate looped B=200
host-side NSB entropy fits per target, ~7 ms each at K=16, and cells cost 30–98 s per
replicate at production size; ~270 chunks × 16 workers ran for most of a day on a shared
20-core box. No profile at production shapes was taken before launch.)
