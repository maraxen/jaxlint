---
name: developing-composable-jax
description: Use when Equinox __call__ methods are growing complex with branching paths, when similar modules are diverging, or when one operation needs several interchangeable implementations. Model-structure patterns only; batching/axis planning, pipeline stages and scan-vs-while loop selection live in xtrax (using-xtrax).
when_to_use: 'Triggers: composable-jax, module-diverging, branching __call__, swappable implementation.'
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

**Why:** Tuple loops compose cleanly with `jax.vmap` and `jax.jit`, and the loop body is identical for every layer.

**Depth caveat:** a Python loop over `self.layers` is unrolled at trace time, so the jaxpr and the compile time grow
linearly with depth. For a deep stack of *identical* layers, build the layers with `eqx.filter_vmap` so their
parameters are stacked along a leading axis, then iterate with `jax.lax.scan` over that axis: one traced body
regardless of depth. Keep the tuple loop for short or heterogeneous stacks.

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


## SECTION 5: Pattern 4 — Protocol injection & ABC inheritance (MEDIUM-HIGH COMMITMENT)

Define a protocol for a swappable algorithm slot, or an ABC when the implementations share base logic. The caller holds the slot and never branches on which implementation it has.

**Simple protocol approach (duck typing):**

```python
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp


class Activation(Protocol):
    def __call__(self, x: jax.Array) -> jax.Array: ...


class ScaledTanh(eqx.Module):
    scale: float = eqx.field(static=True)

    def __call__(self, x):
        return jnp.tanh(x) * self.scale


class LearnedSwish(eqx.Module):
    beta: jax.Array

    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)


class Block(eqx.Module):
    linear: eqx.nn.Linear
    activation: Activation  # any Module satisfying the protocol; Block never branches on it

    def __call__(self, x):
        return self.activation(self.linear(x))
```

**ABC approach (shared base implementation & composite variants):**

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

Note: `eqx.Module, ABC` multiple inheritance works correctly. The integrator loop calls `constraint.apply_positions(state)` without knowing which subclass it holds.

**Caution on protocols:** `@runtime_checkable` on a single `__call__` method matches ANY callable. Don't use `isinstance(x, Activation)` for behavioral guarantees, because it's too broad. Prefer static type checking (ty/mypy/pyright) for protocol conformance.

**When to use:** 3+ interchangeable implementations of the same operation, or when the caller should be agnostic to the strategy. Use ABC when you have shared base logic or composite variants; use Protocol when you just want duck typing.

**Skip when:** You have only 1–2 implementations with no third planned.

**In xtrax:** the pipeline-stage protocols (`Fuse`, `Tap`, `Sink` in `xtrax.stages.boundaries`, and `FuseFn` in `xtrax.stages.protocols`) are this pattern applied to stage boundaries. Use them rather than re-declaring your own when you build on xtrax.

---

## SECTION 6: Patterns xtrax implements: batching, stage bundles, loop selection

Three composability problems recur often enough that [xtrax](https://github.com/maraxen/xtrax) implements them as library primitives. If you use xtrax, use these; its `using-xtrax` skill is the reference. If you don't, the ideas still apply, but write them as small local helpers rather than a framework.

| Problem | Idea | xtrax primitive |
|---|---|---|
| Variable-length inputs and memory-bounded batching | Bucket lengths, pad to the bucket, carry a mask; per axis, choose `vmap` vs tiled `lax.map` from a memory budget (heterogeneous axes always tile) | `AxisSpec`, `BatchPlanner` / `MemoryBudget`, `ChunkedMap`, length bucketing (`using-xtrax`, `references/length-bucketing.md`) |
| Several interchangeable pipeline components (encode, decode, sample) | Resolve the components once into a bundle/plan on the host; the traced code receives the resolved callables and never branches on mode | `StageBundle`, `RunSpec`, `xtrax.stages` (`using-xtrax`, `references/run.md`) |
| Scan vs while loop (training vs inference) | `lax.scan` when you differentiate through the loop; `lax.while_loop` for inference-only loops (early stop, and reported large codegen savings on GPU for long autoregressive decodes) | `CarrySpec(collect_outputs=True)` → `Scan`; `CarrySpec(collect_outputs=False, cond=...)` / `WhileCarry` → one `WhileOp` (`using-xtrax`, `references/adoption.md`, "Inference-only loops") |

The general JAX semantics of the scan/while choice (differentiability, termination, why IR size is not the explanation) are in `using-jax`, under "lax.scan vs lax.while_loop".

---
