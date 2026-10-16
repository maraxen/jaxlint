---
name: using-optax
description: Use when defining, initializing, composing, or debugging Optax optimizers — GradientTransformation usage, custom optimizers, learning rate schedules, gradient clipping, or weight decay.
---
# using-optax

## When to Use This Skill

Activate this skill when setting up, composing, customizing, or debugging optimization pipelines using **Optax** in JAX or Equinox projects. Triggers include:
- Initializing or updating standard optimizers (`optax.adam`, `optax.adamw`, `optax.sgd`, etc.).
- Defining training steps and applying updates to parameters.
- Implementing learning rate schedules (warmup, decay, or custom schedulers).
- Composing gradient transformations using `optax.chain`, `optax.partition` (multi-transform), or `optax.masked`.
- Applying gradient clipping (`optax.clip_by_global_norm`, `optax.clip_by_value`).
- Configuring weight decay (AdamW or `optax.add_decayed_weights`) with parameter masks.
- Maintaining parameter moving averages using `optax.incremental_update` (EMA).
- Writing custom gradient transformations or optimizers.

---

## Symptom-to-Fix Table (Quick Reference)

Refer to this table first when troubleshooting Optax issues:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TypeError: update() missing 1 required positional argument: 'params'` | Weight decay or adaptive scaling requires current parameters to compute updates | Pass `params` as the third argument to the update function: `tx.update(grads, state, params)` |
| `TypeError: 'GradientTransformation' object is not callable` | Tried to call the optimizer directly like a function (e.g. `tx(grads, state)`) | Use the `update` method: `updates, state = tx.update(grads, state, params)` |
| Weight decay decays biases or normalisation layers | Default weight decay applies to all leaves in the PyTree | Pass `mask=lambda p: jax.tree.map(lambda x: x.ndim != 1, p)` to `optax.adamw` or `optax.add_decayed_weights` |
| Cosine decay schedule decays to end value too early | Mismatch in `decay_steps` calculation; `decay_steps` is the *total* steps (warmup + decay) | Set `decay_steps = total_steps` (do not subtract `warmup_steps` from `decay_steps`) |
| Parameters are not changing during training | Forgot to apply updates to parameters, or optimizer state is not being propagated | Reassign: `params = optax.apply_updates(params, updates)` and ensure `state` is updated and returned |
| `ValueError: transforms mapping ... does not contain label ...` | A leaf value in the parameter label PyTree does not match any key in the `transforms` dictionary of `optax.partition` | Double-check that the labels returned by your `label_fn` exactly match the keys of the `transforms` dictionary |
| `TracerBoolConversionError` when evaluating schedules | Training step is traced with JIT, but step count is tracked as a Python int rather than a JAX array | Let JAX trace the step count as an array and pass it to JIT-compiled update steps |
| EMA weights do not track parameters | Using `optax.incremental_update` with incorrect argument order | Always use `optax.incremental_update(new_tensors=params, old_tensors=ema_params, step_size=tau)` |

---

## Core API: GradientTransformation

In Optax, all optimizers are implemented as stateless **gradient transformations** (`optax.GradientTransformation`). An optimizer consists of two pure functions:
1. `init(params)`: Initializes state (momentum, step count, etc.) based on parameter shapes/dtypes.
2. `update(updates, state, params=None)`: Transforms gradients and returns the new updates and updated state.

### Standard Training Loop Pattern

```python
import jax
import jax.numpy as jnp
import optax

# 1. Define optimizer
optimizer = optax.adam(learning_rate=1e-3)

# 2. Initialize parameters and optimizer state
params = {'w': jnp.ones((10, 2)), 'b': jnp.zeros(2)}
opt_state = optimizer.init(params)

# JIT-compiled training step
@jax.jit
def train_step(params, opt_state, x, y):
    def loss_fn(params):
        preds = x @ params['w'] + params['b']
        return jnp.mean((preds - y) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # 3. Compute updates and new state
    # Note: Pass params as 3rd arg in case weight decay is added later
    updates, opt_state = optimizer.update(grads, opt_state, params)
    
    # 4. Apply updates to parameters
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

---

## Composing Optimizers

Optax allows building custom optimization pipelines by chaining smaller transformations together.

### 1. Sequential Chaining with `optax.chain`

`optax.chain` applies gradient transformations sequentially from left to right.
*Important*: Transformations like gradient clipping or weight decay should typically precede the main optimizer in the chain.

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),      # 1. Clip global gradient norm
    optax.add_decayed_weights(1e-4),      # 2. Add weight decay
    optax.scale_by_adam(),                # 3. Apply Adam scaling (moments)
    optax.scale_by_learning_rate(1e-3)    # 4. Multiply by learning rate
)
```

### 2. Multi-Optimizer Partitioning with `optax.partition`

Use `optax.partition` to apply different optimizers/learning rates to different parts of the parameter tree (e.g., different learning rate for a pre-trained backbone vs. a newly initialized classification head).

```python
import jax

params = {
    'backbone': {'w': jnp.ones((100, 100)), 'b': jnp.zeros(100)},
    'head': {'w': jnp.ones((100, 10)), 'b': jnp.zeros(10)}
}

# 1. Define a mapping dictionary of labels to optimizers
transforms = {
    'backbone_opt': optax.adam(learning_rate=1e-4),
    'head_opt': optax.adam(learning_rate=1e-3)
}

# 2. Define a label tree or a function mapping parameter paths to keys in `transforms`
def label_fn(params):
    return jax.tree.map(
        lambda _: 'backbone_opt', params
    )  # Default all to backbone_opt
    
# Or explicitly construct a matching label tree:
param_labels = {
    'backbone': {'w': 'backbone_opt', 'b': 'backbone_opt'},
    'head': {'w': 'head_opt', 'b': 'head_opt'}
}

# 3. Create the partitioned optimizer
optimizer = optax.partition(transforms, param_labels)
opt_state = optimizer.init(params)
```

### 3. Parameter Freezing with `optax.masked` or `optax.partition`

To freeze specific parameters (prevent them from updating), map them to `optax.set_to_zero()`.

```python
# Option A: Using optax.partition (Recommended for clean parameter structure)
transforms = {
    'train': optax.adam(1e-3),
    'freeze': optax.set_to_zero()
}
param_labels = {
    'backbone': 'freeze',  # Entire backbone subtree is frozen
    'head': 'train'        # Head is updated
}
optimizer = optax.partition(transforms, param_labels)

# Option B: Using optax.masked
mask = {
    'backbone': False,     # False = do not apply optimizer
    'head': True           # True = apply optimizer
}
optimizer = optax.chain(
    optax.adam(1e-3),
    optax.masked(optax.set_to_zero(), mask=mask)
)
```

---

## Learning Rate Schedules

Instead of a static float, you can pass a callable schedule function to most Optax optimizers as the `learning_rate`.

### Cosine Decay with Warmup

The `optax.warmup_cosine_decay_schedule` transitions linearly from `init_value` to `peak_value` over `warmup_steps`, and then decays to `end_value` using a cosine curve over the remaining steps.
*Crucial*: `decay_steps` represents the **total length** of the schedule (including the warmup steps).

```python
total_steps = 10_000
warmup_steps = 1_000

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,  # Must be total steps, NOT (total_steps - warmup_steps)
    end_value=1e-5
)

# Pass the schedule directly as the learning rate
optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
```

---

## Weight Decay Discipline

Standard L2 regularization (adding an $L2$ penalty to the loss function) does not behave as intended when using adaptive optimizers like Adam. Always prefer decoupled weight decay (`optax.adamw` or `optax.add_decayed_weights`).

### Exclude Biases and 1D Normalization Weights

To prevent overfitting without decaying biases or normalization parameters (which can cause underfitting), use a mask.

```python
# Define weight decay mask: only decay parameters with ndim != 1 (keeps biases & LayerNorm scales safe)
wd_mask = lambda params: jax.tree.map(lambda x: x.ndim != 1, params)

optimizer = optax.adamw(
    learning_rate=1e-3,
    weight_decay=1e-4,
    mask=wd_mask
)
```

---

## Moving Average Weight Tracking (EMA)

Maintaining an Exponential Moving Average (EMA) of model parameters is a standard method to stabilize evaluation. Use `optax.incremental_update` for Polyak averaging.

```python
# Initialize EMA weights with the same structure/values as training parameters
ema_params = jax.tree.map(jnp.copy, params)

# In the training loop:
for step in range(num_steps):
    params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
    
    # Update EMA parameters incrementally (e.g., decay rate 0.999 => step_size = 0.001)
    ema_params = optax.incremental_update(
        new_tensors=params,
        old_tensors=ema_params,
        step_size=0.001
    )
```

---

## Custom Gradient Transformations

You can write custom gradient transformations by returning a `GradientTransformation` NamedTuple.

```python
from typing import NamedTuple

class ClipAndScaleState(NamedTuple):
    count: jax.Array

def clip_and_scale(clip_threshold: float, scale_factor: float):
    def init_fn(params):
        del params
        return ClipAndScaleState(count=jnp.zeros((), dtype=jnp.int32))
        
    def update_fn(updates, state, params=None):
        # Clip updates element-wise
        clipped_updates = jax.tree.map(
            lambda g: jnp.clip(g, -clip_threshold, clip_threshold), 
            updates
        )
        # Apply scaling
        scaled_updates = jax.tree.map(
            lambda g: g * scale_factor, 
            clipped_updates
        )
        new_state = ClipAndScaleState(count=state.count + 1)
        return scaled_updates, new_state
        
    return optax.GradientTransformation(init_fn, update_fn)
```

If your transformation needs access to additional runtime arguments (e.g. current loss value), subclass/use `optax.GradientTransformationExtraArgs` instead.
