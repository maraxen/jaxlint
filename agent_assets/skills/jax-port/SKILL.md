---
name: jax-port
description: Use when porting a numerical algorithm, model, or training loop from NumPy, PyTorch, or TensorFlow to JAX — including vendoring a reference implementation, decomposing math before coding, structuring graded parity tests, defining data contracts with jaxtyping, porting PyTorch nn.Module to Equinox, converting stateful PRNG to jax.random, handling dynamic shapes with padding/masking, and managing the JIT boundary for logging and data loading.
---

# JAX Porting

## Overview

Porting an algorithm to JAX is a staged, outside-in process: **understand the math first, vendor a clean reference, build pure functions with typed contracts, then verify each stage against the reference before JIT-ing.** Skip any stage and silent numerical bugs or tracing failures appear much later, when they are hardest to diagnose.

This skill covers the full lifecycle: vendoring → decomposition → pure functions + data contracts → JIT conversion → graded parity testing → training-loop JIT boundary. For model ports from PyTorch, also covers Equinox module porting and stateful→stateless PRNG conversion.

**Required sub-skills for quality gates:**
- `using-jax` — JIT mental model, PRNG discipline, sharp bits
- `using-equinox` — module porting, `eqx.partition`, `filter_jit`
- `developing-composable-jax` — YAGNI gate, batch/axis management
- `using-jaxlint` — run `jaxlint check` as a gate at each phase
- `test-driven-development` — write parity tests before JAX implementation
- `verification-before-completion` — require green parity suite before declaring done

---

## Phase 0 — Vendor the Reference

Before touching JAX, pin a clean, importable copy of the reference implementation.

```
port/
  reference/
    algo.py          # unmodified source — NumPy, PyTorch, or SciPy
    __init__.py      # re-exports only; never add logic here
  jax_port/
    algo.py          # JAX port lives here
  tests/
    test_parity_algo.py   # diff-tests reference vs JAX port
```

Rules:
- Add `# REFERENCE: DO NOT MODIFY` at the top of every reference file.
- Import the reference module from `tests/` only — never from production code.
- Pin the exact commit/version in `pyproject.toml` or a comment in the reference header.
- Vendor (copy source) when upstream has heavy production-incompatible dependencies; install as dev-only when the package is lightweight.
- Delete the `reference/` subtree only after all parity tiers pass.

---

## Phase 1 — Decompose Before Coding

Write the math before writing any JAX. Three layers, in order:

**1a. Natural language** — one paragraph describing what the algorithm computes and why. No symbols yet. This forces understanding.

**1b. Mathematical notation (LaTeX)** — each operation as a formula in a `Math:` docstring block:

```
Math:
    Given $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{L \times H}$:

    $$\text{scores}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{H}}, \quad
      \mathbf{a} = \text{softmax}(\text{scores})$$
```

**1c. Pseudocode** — write the algorithm as a `Pseudocode:` docstring section (jaxlint canonical section order). This is your executable spec:

```
Pseudocode:
    scores = q @ k.T / sqrt(H)       # [L, L]
    if mask: scores[~mask] = -inf
    weights = softmax(scores, axis=-1)
    return weights @ v               # [L, H]
```

Only proceed to code once the pseudocode matches the reference implementation's logic under manual inspection.

---

## Phase 2 — Pure Functions + Data Contracts

Convert pseudocode to JAX using `jnp` only — no `jit`, `vmap`, or `grad` yet.

**Data contracts with jaxtyping** (scored positively by jaxlint rubric):

```python
from jaxtyping import Array, Float, Bool
import jax
import jax.numpy as jnp

def scaled_dot_product_attention(
    q: Float[Array, "seq heads"],
    k: Float[Array, "seq heads"],
    v: Float[Array, "seq heads"],
    mask: Bool[Array, "seq"] | None = None,
) -> Float[Array, "seq heads"]:
    """
    Scaled dot-product attention for a single sequence.

    Math:
        $$\text{scores} = QK^T / \sqrt{H}, \quad \mathbf{a} = \text{softmax}(\text{scores})$$

    Pseudocode:
        scores = q @ k.T / sqrt(heads)
        if mask: scores[~mask] = -inf
        weights = softmax(scores, axis=-1)
        return weights @ v
    """
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum("sh,th->st", q, k) * scale
    if mask is not None:
        scores = jnp.where(mask[None, :], scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1) @ v
```

**Note on the double-where trick (grad safety):** When a masked branch contains a function with NaN gradients at the masked values (e.g. `jnp.sqrt(x)` at `x≤0`, `jnp.log(x)` at `x≤0`), use the double-where pattern to prevent NaN gradient propagation:

```python
# WRONG: gradient at masked positions is NaN even though output is masked
y = jnp.where(mask, jnp.sqrt(x), 0.0)

# CORRECT: replace bad values before op, then mask the result
safe_x = jnp.where(mask, x, jnp.ones_like(x))   # 1.0 has safe sqrt gradient
y = jnp.where(mask, jnp.sqrt(safe_x), 0.0)
```

**Contract rules:**
- Every array argument gets a `jaxtyping` annotation with named axes.
- Static (non-traced) arguments use plain Python types (`int`, `str`, `bool`). In Equinox modules, mark with `eqx.field(static=True)`.
- Use `@dataclass(frozen=True, slots=True)` for structured inputs passed as PyTree containers.
- No `isinstance()` checks on JAX arrays — trust type annotations.

**Run `jaxlint check` here.** Resolve all JL-rule errors before proceeding.

---

## Phase 3 — Dynamic Shapes: Padding and Masking

JIT requires static shapes. Variable-length inputs (sequences, graphs, point clouds) must be padded to a fixed shape at the host level before entering `jit`.

```python
def pad_to_length(x: np.ndarray, length: int, pad_value=0.0):
    """Pad sequence to static length; return array + boolean mask."""
    n = x.shape[0]
    padded = np.pad(x, [(0, length - n)] + [(0, 0)] * (x.ndim - 1), constant_values=pad_value)
    mask = np.arange(length) < n  # True = real token, False = padding
    return padded, mask
```

Inside the kernel, carry the mask and apply it everywhere sequences are reduced:

```python
# Mask logits before softmax (prevents padding from attending to real tokens)
scores = jnp.where(mask[None, :], scores, jnp.finfo(scores.dtype).min)

# Mask losses before mean (prevents padding from diluting the loss)
loss = jnp.where(mask, per_token_loss, 0.0).sum() / mask.sum()
```

**Bucketing:** For throughput, group sequences into fixed-length bins (e.g. 128, 256, 512) rather than padding everything to the global maximum. Recompilation occurs once per bucket.

**What you cannot do inside `jit` on traced values:**
- `x[:n]` where `n` is traced → use `dynamic_slice(x, (0,), (n,))` or pre-pad
- `jnp.nonzero(mask)` → returns dynamic shape; use `jnp.where` with fixed shape instead

---

## Phase 4 — Equinox Module Porting (PyTorch → JAX)

PyTorch `nn.Module` allocates parameters via `__init__`; Equinox modules are immutable PyTrees with explicit parameter fields.

```python
# PyTorch original
class MyLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.training = True

    def forward(self, x):
        return F.dropout(x @ self.weight + self.bias,
                         p=0.1, training=self.training)

# Equinox equivalent
import equinox as eqx

class MyLayer(eqx.Module):
    weight: Float[Array, "d_in d_out"]
    bias: Float[Array, "d_out"]

    def __init__(self, d_in: int, d_out: int, key: jax.Array):
        self.weight = jax.random.normal(key, (d_in, d_out)) * (d_in ** -0.5)
        self.bias = jnp.zeros(d_out)

    def __call__(
        self,
        x: Float[Array, "seq d_in"],
        key: jax.Array,  # dropout key — explicit, not global state
        inference: bool = False,  # replaces self.training
    ) -> Float[Array, "seq d_out"]:
        x = x @ self.weight + self.bias
        return eqx.nn.Dropout(p=0.1)(x, key=key, inference=inference)
```

**Key translation rules:**
- `nn.Parameter` → typed field; init via `jax.random.normal` / `jax.random.key`
- `self.training` → explicit `inference: bool` argument (static field or call arg)
- Dropout and other stochastic layers → require a `key` argument; split before each call
- `model.train()` / `model.eval()` → pass `inference=True/False` to `__call__`
- `optimizer.step()` → `eqx.apply_updates(model, updates)` after `optax` gradient step

**Checkpoint translation (params only):**
```python
# Extract arrays for checkpointing (exclude static metadata)
arrays, static = eqx.partition(model, eqx.is_array)
# Save arrays; restore and recombine:
model = eqx.combine(restored_arrays, static)
```

---

## Phase 5 — PRNG: Stateful → Stateless

This is the most common source of parity bugs when porting from PyTorch. JAX PRNG is **explicit and stateless** — every stochastic call requires a fresh key derived from the parent.

```python
# PyTorch: global state, implicit
torch.manual_seed(42)
dropout_out = F.dropout(x, p=0.1)  # consumes global RNG state

# JAX: explicit key threading
key = jax.random.key(42)
key, drop_key = jax.random.split(key)
dropout_out = eqx.nn.Dropout(p=0.1)(x, key=drop_key)
```

**Key rules:**
- Never reuse a key. Split before every stochastic call.
- Thread keys through `scan` via the carry: `(state, key)`.
- For batched stochasticity under `vmap`, use `jax.random.fold_in` to derive per-sample keys: `keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(batch_size))`.
- Same integer seed ≠ same numbers between PyTorch and JAX — different algorithms. For parity tests: **disable all stochasticity in both reference and JAX port** during value-parity tests. Test stochasticity separately (distribution shape, moments).

---

## Phase 6 — JIT + lax Primitives

Add `@jax.jit` and replace Python control flow. Apply `vmap` and `grad` only after `jit` is stable.

Common replacements:

| Python pattern | JAX replacement |
|---|---|
| `if traced_bool:` | `jax.lax.cond(cond, true_fn, false_fn)` |
| `for i in range(dynamic_n):` | `jax.lax.fori_loop` or `jax.lax.scan` |
| `while condition(x):` | `jax.lax.while_loop` |
| `x[i] = v` | `x.at[i].set(v)` — never immediately subscript result (JL001) |
| `print(loss)` | `jax.debug.print("{loss}", loss=loss)` |
| side-effectful logging | `jax.experimental.io_callback` (see Phase 7) |

**Float precision — production vs parity tests:**
- **Production kernels:** `float32` or `bfloat16`. Avoid `float64` (JL010 — doubles memory, slows GPU).
- **Parity tests only:** enable x64 at module scope:
  ```python
  # conftest.py or top of parity test module
  import jax
  jax.config.update("jax_enable_x64", True)
  ```
- JL010 exempts test files. The contradiction is intentional: production uses float32; parity tests need float64 to catch precision drift separate from algorithm drift.

**Run `jaxlint check` again** after JIT conversion.

---

## Phase 7 — Graded Parity Testing

Write parity tests **before** writing the JAX implementation (TDD). Escalate tier by tier; do not proceed to the next tier until the current one is green on CPU.

**If you use [xtrax](https://github.com/maraxen/xtrax):** its `xtrax.testing` package is a framework-neutral parity harness: an independent oracle, `NegativeControl` checks that must fail, distributional comparison for samplers, and knob coverage. See `using-xtrax`, `references/parity.md`. It also records a ported module's upstream origin with `__provenance__` (`xtrax.provenance`). Use those rather than re-building the tiers below by hand. The tiers still describe what a port has to prove.

### Tier 1 — Dtype + Shape (zero tolerance)
```python
def test_parity_dtype_shape():
    ref_out = reference_algo(x_np)
    jax_out = np.array(jax_algo(jnp.array(x_np)))
    assert ref_out.shape == jax_out.shape, f"{ref_out.shape} vs {jax_out.shape}"
    assert ref_out.dtype == jax_out.dtype, f"{ref_out.dtype} vs {jax_out.dtype}"
```

### Tier 2 — Numerical parity, float64
```python
# Set at module scope in test file — NOT as a context manager (no such API)
jax.config.update("jax_enable_x64", True)

def test_parity_float64():
    x64 = x_np.astype(np.float64)
    ref = reference_algo(x64)
    jax_out = np.array(jax_algo(jnp.array(x64, dtype=jnp.float64)))
    np.testing.assert_allclose(jax_out, ref, rtol=1e-10, atol=1e-10)
```

### Tier 3 — Float32 parity (production dtype)

Tolerance depends on reduction depth — choose the preset that matches your algorithm:

```python
# Elementwise ops (no reductions): tight tolerances
ELEMENTWISE = dict(rtol=1e-6, atol=1e-6)
# Reductions (softmax, layernorm, mean over long axis): looser
REDUCTION = dict(rtol=1e-4, atol=1e-4)
# Matmul on GPU (TF32 vs FP32 on CPU): set highest matmul precision in test
# jax.default_matmul_precision('highest') or env JAX_DEFAULT_MATMUL_PRECISION=highest

def test_parity_float32():
    ref = reference_algo(x_np.astype(np.float32))
    with jax.default_matmul_precision("highest"):
        jax_out = np.array(jax_algo(jnp.array(x_np, dtype=jnp.float32)))
    np.testing.assert_allclose(jax_out, ref, **REDUCTION)
```

**XLA numerical caveats:**
- bfloat16 on TPU uses stochastic rounding in some matmul paths — parity against CPU fp32 reference will fail with tight tolerances; use `rtol=1e-2` or compare in float32.
- Fast-math reductions may reorder ops → ~1 ULP drift per reduction step.
- Default GPU matmul precision is TF32 (`jax.default_matmul_precision('default')`); force `'highest'` in parity tests for fair comparison.

### Tier 4 — Gradient parity (if differentiable)
```python
def test_parity_gradients(atol=1e-4):
    # Disable stochasticity in both reference and JAX for clean gradient comparison
    ref_loss = reference_loss(x_np)
    ref_grads = torch.autograd.grad(ref_loss, ref_params)

    jax_loss_fn = lambda params: jax_loss(params, jnp.array(x_np))
    jax_grads = jax.grad(jax_loss_fn)(jax_params)

    # Build name→array dicts so failures identify which parameter mismatches.
    # param_names must be in the same leaf order for both frameworks — verify this once.
    ref_grad_dict = {name: g.detach().numpy() for name, g in zip(param_names, ref_grads)}
    jax_grad_dict = {name: np.array(g) for name, g in zip(param_names, jax.tree.leaves(jax_grads))}
    for name in ref_grad_dict:
        np.testing.assert_allclose(jax_grad_dict[name], ref_grad_dict[name],
                                   atol=atol, err_msg=f"gradient mismatch: {name}")
```

### Tier 5 — JIT + vmap invariance
```python
def test_jit_stable():
    eager = np.array(jax_algo(x))
    jitted = np.array(jax.jit(jax_algo)(x))
    np.testing.assert_allclose(jitted, eager, atol=1e-6)

def test_vmap_consistent():
    batched = np.array(jax.vmap(jax_algo)(xs))
    sequential = np.stack([np.array(jax_algo(x)) for x in xs])
    np.testing.assert_allclose(batched, sequential, atol=1e-6)
```

**Advancement rule:** CPU must be green before running on GPU/TPU. Tiers 4–5 require Tier 3 to pass first.

---

## Phase 8 — JIT Boundary: Logging and Data Loading

### io_callback vs pure_callback

Use **`jax.experimental.io_callback`** for side-effectful operations (logging, metrics, progress) inside a `jit`-compiled step:

```python
def log_metrics(step, loss, grad_norm):
    wandb.log({"loss": float(loss), "grad_norm": float(grad_norm)}, step=int(step))

@jax.jit
def train_step(state, batch):
    loss, grads = jax.value_and_grad(compute_loss)(state.params, batch)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    # ordered=True: callbacks fire in program order on the host
    jax.experimental.io_callback(log_metrics, None, state.step, loss, grad_norm, ordered=True)
    return state.apply_gradients(grads=grads)
```

**io_callback caveats:**
- `ordered=True` serializes within a single device/replica. Under `pmap` or `shard_map`, callbacks fire once per replica — ordering across replicas is not guaranteed.
- `vmap` of an `ordered=True` callback **raises** (`ValueError: Cannot vmap ordered IO callback`). An unordered callback vmaps, firing once per batch element, which can dominate runtime. Either way, aggregate metrics before the callback, not inside a vmapped function.
- `jax.grad` through an `io_callback` raises (`IO callbacks do not support JVP`). Place it outside the differentiated region, as the `train_step` example does: it logs the loss *after* `value_and_grad`.

Use **`jax.pure_callback`** for pure host-side operations (NumPy/SciPy ops not in JAX):

```python
@jax.jit
def f(x):
    return jax.pure_callback(
        lambda x: scipy.special.expit(np.array(x)).astype(np.float32),
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        x,
        vmap_method="sequential",  # required for vmap; see below
    )
```

| | `pure_callback` | `io_callback` |
|---|---|---|
| Side effects | No | Yes |
| `vmap` | Only with `vmap_method=` (`"sequential"`, `"expand_dims"`, `"broadcast_all"`, …); raises without it | Unordered: yes. `ordered=True`: raises |
| `grad` | Raises; wrap in `jax.custom_jvp` / `custom_vjp` with your own derivative | Raises |
| `ordered=` | N/A | `True` for serial |
| Use for | NumPy/SciPy ops | Logging, metrics |

(Behaviour above verified on jax 0.11.1.)

### Data loading: keep preprocessing outside JIT

All preprocessing (tokenization, augmentation, padding) must happen on host workers — never inside `jit`. Grain batches are numpy arrays; call `jax.device_put` explicitly before the train step.

**Google Grain** (preferred for JAX training):
```python
import grain.python as grain

source = grain.ArrayRecordDataSource(["data/train.array_record"])
sampler = grain.IndexSampler(
    num_records=len(source),
    num_epochs=None,
    shard_options=grain.ShardByJaxProcess(drop_remainder=True),
    shuffle=True, seed=42,
)
loader = grain.DataLoader(
    data_source=source,
    sampler=sampler,
    worker_count=0,        # start with 0; increase after validating correctness
    worker_buffer_size=2,
    operations=[grain.Batch(batch_size=32, drop_remainder=True)],
)
for batch in loader:
    batch = jax.device_put(batch)
    train_state = train_step(train_state, batch)
```

**Grain caveats:**
- Start with `worker_count=0` (synchronous) to validate preprocessing correctness. Increase once parity tests pass.
- `worker_count>0` with CUDA requires `multiprocessing_options` configured to use `spawn` or `forkserver`.
- `ShardByJaxProcess` only shards the data sampler; the train step itself must be sharded separately (`shard_map` / `pmap`) for multi-device training.

**Migration from PyTorch DataLoader:** `DataLoader` → `grain.DataLoader`; `Dataset` → `grain.ArrayRecordDataSource` or `grain.MapDataset`. Audit every `collate_fn` for PyTorch tensor operations — convert to NumPy.

---

## Common Mistakes (port-specific)

| Mistake | Fix |
|---|---|
| Dtype mismatch: reference float64, JAX float32 | Assert dtypes as Tier 1; enable x64 only in test module |
| PRNG parity: same seed ≠ same numbers across frameworks | Disable all stochasticity for value-parity tests |
| Reference module imported in production code | Reference imports in `tests/` only |
| Skip a parity tier under time pressure | Tier skipping hides bugs that surface during training; enforce the gate |
| `jax.experimental.enable_x64()` used as context manager | No such API; use `jax.config.update("jax_enable_x64", True)` at module scope |
| `io_callback` inside `jax.grad` | Move logging outside the differentiated region |
| Grain `worker_count>0` without testing `worker_count=0` first | Start synchronous; scale up after correctness confirmed |

---

## Checklist

- [ ] Phase 0: Reference vendored; `# REFERENCE: DO NOT MODIFY` headers in place
- [ ] Phase 1: Natural language + LaTeX `Math:` + `Pseudocode:` docstring sections written
- [ ] Phase 2: jaxtyping annotations on all array args; double-where used where needed; `jaxlint check` passes
- [ ] Phase 3: Dynamic shapes handled with padding + mask; bucket strategy defined if variable-length
- [ ] Phase 4 (models): Equinox module translated; `eqx.field(static=True)` for non-array fields; stochastic layers take explicit `key`
- [ ] Phase 5 (stochastic): PRNG converted to explicit key threading; stochasticity disabled in parity tests
- [ ] Phase 6: Python control flow replaced with lax; `jaxlint check` passes
- [ ] Phase 7: Parity Tiers 1–3 green on CPU; Tiers 4–5 where applicable; matmul precision set in tests
- [ ] Phase 8 (training loops): `io_callback` for metrics; `pure_callback` for custom ops; data loading outside JIT; Grain `worker_count=0` validated first
- [ ] Reference module deleted after all tiers pass
