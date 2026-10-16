---
name: developing-composable-jax
description: Use when Equinox __call__ methods are growing complex with branching paths, when similar modules are diverging, when vmap/memory management requires axis-aware dispatch, or when an inference pipeline has multiple interchangeable decode/sample modes.
---

# developing-composable-jax

## SECTION 1: YAGNI gate — when to reach for composability

Reach for composability ONLY when branching or duplication has ALREADY happened at least twice:

- `if/elif/else` branching inside `__call__` selecting different compute paths (not just masking)
- The same `__call__` body is used with feature flags toggled in 3+ call sites
- A method is growing past ~50 LOC with nested closures or nested vmaps
- Multiple subclasses would differ only by one parameter to `__call__`

Do NOT reach for composability when:
- You have 2 variants with no third planned, AND they don't differ in fields or call signatures
- A function is long because data flows through it linearly (no branching)
- The code is new and patterns haven't repeated yet
- You have only a single inference mode
- You have one decoder or one encode path
- The module structure is new/unproven

**Prefer duplication over a wrong abstraction.** Write the pattern once more before abstracting.

**The YAGNI gate is the most important section.** Composability is only warranted when complexity is already high. Premature abstraction kills simplicity and creates maintenance burden.

---

## SECTION 2: Pattern 1 — Class variants over runtime branching (LOW COMMITMENT)

Instead of an `if physics_enabled:` inside `__call__`, define `Encoder` and `PhysicsEncoder` as separate `eqx.Module` subclasses. The caller selects the class at model init time. Neither class has branching.

```python
# Before: branching in __call__
class Encoder(eqx.Module):
    def __call__(self, x, *, physics_features=None):
        if physics_features is not None:
            x = self.physics_proj(physics_features)
        ...

# After: variant classes
class Encoder(eqx.Module):
    layers: tuple[EncoderLayer, ...]
    def __call__(self, x): ...

class PhysicsEncoder(eqx.Module):
    layers: tuple[EncoderLayer, ...]
    physics_projection: eqx.nn.Linear
    def __call__(self, x, physics_features): ...

# Caller selects at init:
encoder = PhysicsEncoder(...) if use_physics else Encoder(...)
```

**When to use:** 2+ stable variants with different fields or different call signatures. This pattern eliminates branching from the hot path and makes the variant contract explicit at type level.

**Skip when:** You have fewer than 2 stable variants.

---

## SECTION 3: Pattern 2 — Layer tuple stacks (LOW COMMITMENT)

Compose repeated layers as `layers: tuple[LayerType, ...]` and loop in `__call__`. No indexing, no branching — the loop body is identical for every layer.

```python
class Encoder(eqx.Module):
    layers: tuple[EncoderLayer, ...]
    node_features_dim: int = eqx.field(static=True)  # static: not a JAX array
    
    def __call__(self, node_features, edge_features, ...):
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, ...)
        return node_features, edge_features
```

Static ints used inside `__call__` can trigger spurious recompilation when their values change between calls. Mark dimension constants with `eqx.field(static=True)` if you need them. For layer counts, use `len(self.layers)` instead.

**Why:** Tuple loops compose cleanly with `jax.vmap` and `jax.jit`. The same bytecode runs for every layer; JAX can fuse the entire stack.

**Skip when:** There's only one layer type or the stack depth won't vary.

---

## SECTION 4: Pattern 3 — Functional pure composition (LOW COMMITMENT)

Loss functions, metrics, and transforms should be pure functions (no `self`, no mutable state), composed via explicit parameters or closures over constant data.

```python
def cross_entropy_loss(logits, targets, mask, label_smoothing=0.0):
    """Pure: no self, no state."""
    ...

# In a training step (not directly JITted):
def make_loss_fn(x, sequence, mask):
    """Create a loss function with explicit closure parameters."""
    def loss_fn(model):
        logits_batch = jax.vmap(model)(x)
        def batch_loss(logits, seq, msk):
            return cross_entropy_loss(logits, seq, msk)
        return jnp.mean(jax.vmap(batch_loss)(logits_batch, sequence, mask))
    return loss_fn

(loss, aux), grads = eqx.filter_value_and_grad(make_loss_fn(x, sequence, mask), has_aux=True)(model)
```

**When to use:** Any function that doesn't need model weights is a pure function candidate. Pure functions compose well with higher-order transforms and are easier to test in isolation.

**Skip when:** The computation genuinely needs model state — keep it as a method.

---

## SECTION 5: Pattern 4 — Batch/Axis management (bucketing, padding, masking) (MEDIUM COMMITMENT)

JAX requires static shapes for `vmap`. When data has variable lengths (e.g., proteins of different residue counts), solve this at the host level before entering JIT:

**Step 1 — Bucket** variable-length inputs into fixed-length bins (e.g., lengths ≤ 256, ≤ 512, ≤ 1024). All inputs in a bucket get padded to the bucket's fixed length.

**Step 2 — Pad** to the bucket length, adding zeros for positions. Keep a boolean `mask` (True = real, False = padding).

**Step 3 — Mask** inside kernels: multiply attention scores and losses by the mask to exclude padded positions from computations.

**Step 4 — Tile with safe_map** when a batch axis can't fit in memory even with fixed shapes. `safe_map` dispatches to `vmap` (batch_size=0) or `jax.lax.map` with tiling (batch_size>0):

```python
def safe_map(f, xs, batch_size=None):
    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        raise ValueError("xs must not be empty")
    n = leaves[0].shape[0]
    if batch_size is None or batch_size == 0 or n <= batch_size:
        return jax.vmap(f)(xs)
    return jax.lax.map(f, xs, batch_size=batch_size)
```

**BatchPlanner pattern**: for multi-axis dispatch (structures × samples × temperatures × noises), use a planner that decides per-axis vmap vs tiling based on device memory budget:
- Heterogeneous axes (variable shapes) → always tile
- Homogeneous axes → try vmap first, degrade to tiling if budget exceeded
- Nest `safe_map` calls: `structures → noises → temperatures → samples`

**When to use:** Protein/molecular data with variable lengths; multi-sweep inference (samples × temperatures × noises). This pattern keeps JAX's static shape requirements while supporting truly variable data at the Python level.

**Skip when:** All inputs are already the same shape; padding adds overhead and bug surface.

---

## SECTION 6: Pattern 5 — Protocol injection & ABC inheritance (MEDIUM-HIGH COMMITMENT)

Define protocols for swappable algorithm slots, or use ABC when you need shared base implementation. Implementations satisfy the protocol via duck typing or inheritance; callers never branch on which implementation is present.

**Simple protocol approach (duck typing):**

```python
from typing import Protocol

class FuseFn(Protocol):
    def __call__(self, per_item, bias=None): ...

class ArithmeticMeanLogits(eqx.Module):
    weights: jax.Array
    def __call__(self, per_item, bias=None):
        return jnp.average(per_item, axis=0, weights=self.weights)

class GeometricMeanLogits(eqx.Module):
    weights: jax.Array
    def __call__(self, per_item, bias=None):
        return jnp.exp(jnp.average(jnp.log(per_item + 1e-8), axis=0, weights=self.weights))

# Inference loop — never knows which FuseFn it has:
fused = stage_set.logit_transform(multi_state_logits)
```

**ABC approach (shared base implementation & composite variants):**

When you need shared base implementation or a Composite variant:

```python
from abc import ABC, abstractmethod

class ConstraintAlgorithm(eqx.Module, ABC):
    @abstractmethod
    def apply_positions(self, state): ...
    @abstractmethod
    def apply_velocities(self, state): ...

class SETTLEConstraint(ConstraintAlgorithm): ...
class NullConstraint(ConstraintAlgorithm): ...

class CompositeConstraint(ConstraintAlgorithm):
    constraints: tuple[ConstraintAlgorithm, ...]
    def apply_positions(self, state):
        for c in self.constraints:
            state = c.apply_positions(state)
        return state
```

Note: `eqx.Module, ABC` multiple inheritance works correctly. The integrator loop calls `constraint.apply_positions(state)` without knowing which subclass it holds; new constraint types add a subclass; the loop never changes.

**Caution on protocols:** `@runtime_checkable` on a single `__call__` method matches ANY callable. Don't use `isinstance(x, FuseFn)` for behavioral guarantees — it's too broad. Prefer static type checking (mypy/pyright) for protocol conformance.

**When to use:** 3+ interchangeable implementations of the same operation, or when the caller should be agnostic to the strategy. Use ABC when you have shared base logic or composite variants; use Protocol when you just want duck typing.

**Skip when:** You have only 1–2 implementations with no third planned.

---

## SECTION 6: Pattern 6 — StageSet / Bundle / Plan (resolve-once) (HIGH COMMITMENT)

For inference pipelines with multiple interchangeable components, bundle resolved implementations once and reuse.

```python
from typing import NamedTuple, Callable

# Define reusable component bundle
class InferenceComponents(NamedTuple):
    encode_fn: Callable
    driver: Callable
    stage_set: "StageSet"

# StageSet: typed container of composable pipeline stages
class StageSet(eqx.Module):
    logit_transform: FuseFn | None = None
    decode_step: ConditionalDecodeStep | UnconditionalDecodeStep | None = None
    sample_step: Any | None = None

# InferencePlan: host-side dataclass, NOT eqx.Module (not JIT-traced)
class InferencePlan:
    """Host-side plan; holds model and pipeline components."""
    model: Any
    components: InferenceComponents
    
    def sample(self, bundle, key, config):
        """Use sample_step path."""
        enc = self.components.encode_fn(bundle, key, config)
        return self.components.driver(
            self.model, key, enc, ..., self.components.stage_set.sample_step
        )
    
    def score(self, bundle, key, config):
        """Use decode_step path."""
        enc = self.components.encode_fn(bundle, key, config)
        return self.components.driver(
            self.model, key, enc, ..., self.components.stage_set.decode_step
        )
```

`.sample()` and `.score()` visibly select different driver paths via `stage_set`, making the pipeline contract explicit. `InferencePlan` is a host-side dataclass (NOT `eqx.Module`) — it holds the model and pipeline components but is not JIT-traced itself.

**When to use:** Inference pipelines with 2+ decoding modes (sample/score/conditional/unconditional) that share an encoder. Resolving components once and bundling them avoids repeated instantiation and makes the pipeline contract explicit.

**Skip when:** Single inference mode, fewer than 2 swappable components, or a function with 3 args would do.

---

## SECTION 7: Pattern 7 — AutoregressiveMode: lax.while_loop for inference, lax.scan for training (HIGH IMPACT)

**Context:** Some autoregressive decoders must be trained end-to-end *and* run at inference time with different loop needs:
- Training needs `lax.scan`, which is reverse-mode differentiable.
- Inference may want `lax.while_loop`, for an early stop on EOS, or because a measured compile time is lower on that workload.

At `unroll=1` both primitives lower to a single `stablehlo.while` with one body (verified on jax 0.11.1). So the choice is about differentiability and termination, not IR size.

**Observation, not a rule (prxteinmpnn, Blackwell SM120, bucket=512, B=16):** one decoder took 30–40 min to cold-compile with `lax.scan(unroll=1)` and 6.5 s with `lax.while_loop`. The lowering comparison above does not explain the gap. If you see something similar, measure compile seconds for both variants on your own workload (`jax_log_compiles`, or `xtrax.profiling.count_backend_compiles`) before adopting this pattern for compile time alone.

**The pattern:** define an `AutoregressiveMode` config that selects the loop primitive, then dispatch inside the module's decode method.

```python
import dataclasses
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
from typing import NamedTuple


@dataclasses.dataclass(frozen=True)
class AutoregressiveMode:
    """Select loop primitive based on gradient requirements.

    inference_only=True  → lax.while_loop (can stop early; not reverse-differentiable)
    inference_only=False → lax.scan        (differentiable, training-safe)
    """
    inference_only: bool = False


class ARCarry(NamedTuple):
    """Bundled carry for while_loop — all mutable state in one struct."""
    step: jax.Array        # scalar int, current decode position
    outputs: jax.Array     # (seq_len, d_out) output buffer
    state: jax.Array       # recurrent state / KV cache prefix


class AutoregressiveDecode(eqx.Module):
    step_fn: eqx.Module     # one decode step: (token, state) → (token, state)
    max_len: int = eqx.field(static=True)

    def __call__(
        self,
        init_token: jax.Array,
        init_state: jax.Array,
        *,
        mode: AutoregressiveMode = AutoregressiveMode(),
    ) -> jax.Array:
        if mode.inference_only:
            return self._while_loop_decode(init_token, init_state)
        else:
            return self._scan_decode(init_token, init_state)

    def _while_loop_decode(self, init_token, init_state) -> jax.Array:
        """Inference path: lax.while_loop (add an EOS test to the cond for early stop)."""
        n, d = self.max_len, init_token.shape[-1]
        init = ARCarry(
            step=jnp.array(0),
            outputs=jnp.zeros((n, d), dtype=init_token.dtype),
            state=init_state,
        )

        def body(carry: ARCarry) -> ARCarry:
            token_in = jnp.where(carry.step == 0, init_token, carry.outputs[carry.step - 1])
            token_out, new_state = self.step_fn(token_in, carry.state)
            outputs = carry.outputs.at[carry.step].set(token_out)
            return ARCarry(carry.step + 1, outputs, new_state)

        final = lax.while_loop(lambda c: c.step < self.max_len, body, init)
        return final.outputs

    def _scan_decode(self, init_token, init_state) -> jax.Array:
        """Training path: lax.scan → reverse-mode differentiable."""
        def step(carry, _):
            token_in, state = carry
            token_out, new_state = self.step_fn(token_in, state)
            return (token_out, new_state), token_out

        _, outputs = lax.scan(step, (init_token, init_state), None, length=self.max_len)
        return outputs
```

**Caller usage:**
```python
# Inference (while_loop; not differentiable):
tokens = decoder(init_token, init_state, mode=AutoregressiveMode(inference_only=True))

# Training (scan; differentiable):
tokens = decoder(init_token, init_state)  # default mode
loss = cross_entropy(tokens, targets)
grads = eqx.filter_grad(lambda m: cross_entropy(m(init_token, init_state), targets))(decoder)
```

**YAGNI gate:** Only reach for this pattern when:
- The module needs BOTH training (grad) and inference modes, AND
- inference needs something `scan` cannot give: a data-dependent stop (EOS), or a compile-time
  or run-time win you have *measured* on this workload for the `while_loop` variant.

**Skip when:** the loop only ever runs under grad, inference always runs the full fixed length,
or you have not measured a difference between the two variants.

---
