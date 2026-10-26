---
name: using-equinox
description: Use when defining eqx.Module subclasses, using filter_jit/filter_grad, managing static fields, or debugging Equinox-specific tracing issues.
when_to_use: 'Triggers: eqx.Module, filter_jit, filter_grad.'
---
# using-equinox

Equinox (eqx) is Patrick Kidger's pure-functional neural network library for JAX. Use this skill when working with eqx.Module, filter transforms, static fields, or functional model updates.

## When to use

Trigger this skill when:
- Defining custom `eqx.Module` subclasses
- Using `@eqx.filter_jit` or `@eqx.filter_grad` decorators
- Managing static vs. dynamic fields in modules
- Debugging shape errors or tracing issues with non-array config
- Updating model parameters without mutation
- Batching operations over modules with `jax.vmap`

## eqx.Module basics

All Equinox modules subclass `eqx.Module`. Each field in the class becomes a pytree leaf by default (arrays are leaves, non-arrays are treated as static). The `__init__` method assigns all fields, and `__call__` implements the forward pass:

```python
import jax
import equinox as eqx
import jax.numpy as jnp

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        key_w, key_b = jax.random.split(key)
        self.weight = jax.random.normal(key_w, (out_size, in_size)) * 0.01
        self.bias = jnp.zeros(out_size)

    def __call__(self, x):
        return x @ self.weight.T + self.bias
```

## Static fields

Use `eqx.field(static=True)` for non-array configuration like layer sizes, string flags, or booleans. Static fields prevent JAX from tracing through non-array values and enable shape-dependent logic:

```python
class MLP(eqx.Module):
    layers: list
    hidden_dim: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(self, in_dim, out_dim, hidden_dim, depth, key):
        self.hidden_dim = hidden_dim
        self.activation = 'relu'
        keys = jax.random.split(key, depth + 1)
        self.layers = [
            Linear(in_dim, hidden_dim, keys[0]),
            *[Linear(hidden_dim, hidden_dim, keys[i+1]) for i in range(depth - 1)],
            Linear(hidden_dim, out_dim, keys[-1])
        ]
```

Without `static=True`, JAX tries to trace the integer `hidden_dim`, causing shape errors.

## filter_jit and filter_grad

The `@eqx.filter_jit` decorator automatically treats JAX/NumPy arrays as dynamic and everything else as static. Pair it with `@eqx.filter_grad` for differentiation:

```python
@eqx.filter_jit
@eqx.filter_grad
def loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)
```

For returning both loss and gradients, use `eqx.filter_value_and_grad(has_aux=False)`:

```python
@eqx.filter_jit
def make_step(model, optim, opt_state, x, y):
    def loss_fn(model):
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y) ** 2)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
```

Use `has_aux=True` when returning auxiliary data alongside the loss:

```python
def loss_fn(model):
    pred = jax.vmap(model)(x)
    loss = jnp.mean((pred - y) ** 2)
    acc = jnp.mean(pred.argmax(-1) == y.argmax(-1))
    return loss, acc
(loss, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
```

## partition / combine / filter

For explicit control over which fields are dynamic vs. static, use `eqx.partition` and `eqx.combine`. `eqx.is_array` selects array fields; `eqx.is_inexact_array` selects floating-point arrays (useful for excluding integer arrays from gradients):

```python
import functools as ft

params, static = eqx.partition(model, eqx.is_array)

@ft.partial(jax.jit, static_argnums=1)
@jax.grad
def loss(params, static, x, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)
```

Pass the model through `jax.jit` with `static_argnums=1` to mark the static part as a static argument. `eqx.filter(model, eqx.is_inexact_array)` extracts only floating-point leaves for optimizer updates.

## Updating model state

Never mutate module fields directly. Use `eqx.tree_at` for functional updates:

```python
# Wrong
model.layer.weight = new_weight  # Mutates; JAX sees stale value

# Correct
model = eqx.tree_at(lambda m: m.layer.weight, model, new_weight)
```

With optax, use `eqx.apply_updates` to apply accumulated gradients:

```python
updates, opt_state = optim.update(grads, opt_state)
model = eqx.apply_updates(model, updates)
```

## Built-in layers

Equinox provides common neural network layers in `eqx.nn.*`. Prefer these over reinventing them:

- `eqx.nn.Linear(in_features, out_features, *, key)` — fully connected layer
- `eqx.nn.MLP(in_size, out_size, width_size, depth, *, key)` — multi-layer perceptron
- `eqx.nn.Conv2d(in_channels, out_channels, kernel_size, *, key)` — 2D convolution
- `eqx.nn.LayerNorm(shape)` — layer normalisation
- `eqx.nn.Dropout(p)` — dropout layer (use `inference=True` at eval time)

These layers handle weight initialization and shape logic automatically, making them more reliable than custom implementations.

## Inference mode

Use `eqx.nn.inference_mode` to disable stochastic operations (dropout, batch normalization) for evaluation:

```python
# Disable dropout/stochastic layers for eval:
model_eval = eqx.nn.inference_mode(model)

# Or use inference_mode for backward compatibility:
# model_eval = eqx.nn.inference_mode(model, value=False)  # re-enable stochastic
```

This is essential for reproducible evaluation and prevents randomness in predictions.

## __check_init__

Use `__check_init__` to validate invariants after initialization. This method runs automatically after `__init__` and is useful for catching configuration errors early:

```python
class Encoder(eqx.Module):
    weight: jax.Array
    
    def __check_init__(self):
        if self.weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {self.weight.ndim}D")
```

Invariant checks in `__check_init__` prevent downstream shape errors and improve error messages.

## filter_vmap for ensembles

Use `eqx.filter_vmap` to create ensemble models efficiently by vectorizing over initialization:

```python
# Create an ensemble of 8 models with a single vmap
make_ensemble = eqx.filter_vmap(lambda k: eqx.nn.MLP(2, 1, 16, 2, key=k))
ensemble = make_ensemble(jax.random.split(jax.random.PRNGKey(0), 8))
# ensemble is a single Module where arrays have an extra leading axis
```

This batches model initialization and parameters, reducing memory overhead and enabling fast ensemble predictions.

## vmap over a module

JAX's `jax.vmap` automatically batches operations over modules. Batch a single model over input samples:

```python
model = eqx.nn.MLP(2, 1, 16, 2, key=jax.random.PRNGKey(0))
batch_x = jnp.zeros((10, 2))  # 10 samples, 2 features
batch_y = jax.vmap(model)(batch_x)  # Output shape: (10, 1)
```

## Verify your module

Before running your module, check shapes without materialising arrays using `jax.eval_shape`:

```python
# Check output shape without building full arrays:
out_shape = jax.eval_shape(jax.vmap(model), x_batch)

# Run basic pytree check to ensure all fields are arrays or static:
leaves, treedef = jax.tree_util.tree_flatten(model)
```

This helps catch shape mismatches and structural errors early.

## With xtrax

[xtrax](https://github.com/maraxen/xtrax) installs no Equinox; add `equinox` to your own dependencies.

- An `eqx.Module` model enters xtrax as a plain pytree and comes back as the same type.
- xtrax's own classes subclass `xtrax.Struct`, not `eqx.Module`: `isinstance(x, eqx.Module)` is `False` for them. `eqx.tree_at` works on them, and `eqx.field(static=True)` and `xtrax.field(static=True)` both mark a field static.
- `xtrax.model_jit` is the `filter_jit` equivalent: array leaves are traced, every other leaf and the tree structure key the compile cache, so non-array leaves must be hashable. `xtrax.partition` / `xtrax.combine` mirror `eqx.partition` / `eqx.combine`.

The xtrax side is in its `using-xtrax` skill.

## Common gotchas

| Gotcha | Cause | Fix |
|--------|-------|-----|
| `Shape (int,) and () not compatible` after `@eqx.filter_jit` | Forgot `static=True` on int field; JAX tried to trace it | Add `eqx.field(static=True)` to int, str, bool fields |
| `Tracer cannot be cast to JAX array` | Tried to differentiate through a non-inexact array (e.g., integer index) | Use `eqx.is_inexact_array` filter or ensure only floats are in grads |
| Cannot use Python if on a traced boolean | Python control flow depends on traced boolean | Make boolean `static=True` or use JAX conditionals (`jnp.where`) |
| Silent stale parameter values in training loop | Mutated module field directly (e.g., `model.weight = ...`) | Use `eqx.tree_at` or `eqx.apply_updates` for all updates |
| Module doesn't update despite loss decreasing | Created a new model inside `@jax.jit` instead of returning updated model | Return the updated model from `make_step`; don't create inside loop |
