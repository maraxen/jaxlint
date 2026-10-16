---
name: using-orbax
description: Use when saving/restoring model checkpoints in JAX — CheckpointManager setup, save/restore patterns, async safety, and multi-host considerations.
---

# using-orbax

## When to use

Use Orbax checkpointing when your JAX training loop needs periodic saves, crash recovery, model evaluation from checkpoint, or weight export. Standard pattern: initialize CheckpointManager once, save at intervals, restore from latest step on restart.

## CheckpointManager vs raw Checkpointer

Always use `CheckpointManager` in training loops; reserve raw `PyTreeCheckpointer` for one-off saves only. Manager handles versioning, garbage collection (max_to_keep), async safety, and step tracking. Raw Checkpointer is simpler but requires manual cleanup and has no recovery semantics.

## Basic setup

Construct with path and options:

```python
import orbax.checkpoint as ocp

options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(path, options=options)
```

The path directory will contain subdirectories for each checkpoint (one per step). `max_to_keep=3` keeps only the 3 most recent. `save_interval_steps=2` saves only when at least 2 steps have passed since the last save.

## Save pattern

Inside the training loop:

```python
for step in range(num_steps):
    train_state = train_fn(train_state)
    mngr.save(step, args=ocp.args.StandardSave(train_state))

mngr.wait_until_finished()  # CRITICAL: block until all saves complete
```

Saves are async by default. To guarantee the final state is captured at the end of training, use `force=True`:

```python
# At end of training — force save regardless of interval:
mngr.save(final_step, args=ocp.args.StandardSave(state), force=True)
mngr.wait_until_finished()
```

**Always call `wait_until_finished()` after the loop** to flush in-flight async writes. Note: `wait_until_finished()` does NOT trigger a save if the interval was skipped — it only ensures pending saves complete. Use `force=True` to guarantee the final state is captured.

## Restore and crash recovery

Check if a checkpoint exists, then restore into an abstract target (avoiding reinitialisation), and resume:

```python
import jax

def init_fn(rng):
    return init_state

init_args = (jax.random.PRNGKey(0),)

mngr = ocp.CheckpointManager('/tmp/mydir/', options=ocp.CheckpointManagerOptions(save_interval_steps=4))

start_step = 0
if mngr.latest_step() is not None:
    start_step = mngr.latest_step()
    # Prefer: abstract target avoids reinitialising model just to get shapes
    abstract_state = jax.eval_shape(init_fn, *init_args)
    state = mngr.restore(start_step, args=ocp.args.StandardRestore(abstract_state))

for step in range(start_step, num_steps):
    state = train_step(state)
    mngr.save(step, args=ocp.args.StandardSave(state))
```

`mngr.latest_step()` returns None if no checkpoint exists. `StandardRestore` requires a target pytree with matching structure. Use `jax.eval_shape()` with your initialization function to create an abstract target, avoiding reinitialisation overhead. Alternatively, if state is already initialized, pass it directly.

## Equinox model checkpointing

For Equinox models, checkpoint only arrays, not metadata or non-array leaves. Use `eqx.partition()` to split cleanly:

```python
import jax
import equinox as eqx
import orbax.checkpoint as ocp

# Before save: partition arrays from static metadata
params, static = eqx.partition(model, eqx.is_array)

ckptr = ocp.StandardCheckpointer()
ckptr.save(path / '1', args=ocp.args.StandardSave(params))

# After restore: reconstruct with abstract target
abstract_params = jax.eval_shape(lambda: eqx.partition(model, eqx.is_array)[0])
restored_params = ckptr.restore(path / '1', args=ocp.args.StandardRestore(abstract_params))
model = eqx.combine(restored_params, static)
```

Do not attempt to serialize the full Equinox module; it contains non-serializable leaves.

## Multi-host and distributed training

On multi-host setups (e.g., 8 GPU nodes), point all hosts at a shared filesystem path (GCS, NFS, Lustre, etc.). Do NOT use node-local paths and merge manually — this introduces race conditions and silent corruption. Orbax coordinates barrier synchronization internally across all hosts when they share a common checkpoint directory.

## Best checkpoint tracking

Track the best checkpoint by validation metric and restore from it at the end:

```python
options = ocp.CheckpointManagerOptions(
    max_to_keep=3,
    best_fn=lambda metrics: metrics['eval_loss'],
    best_mode='min',
)
mngr = ocp.CheckpointManager(path, options=options)

# In eval loop, pass metrics:
mngr.save(step, args=ocp.args.StandardSave(state), metrics={'eval_loss': eval_loss})

# Restore best checkpoint at the end:
best_step = mngr.best_step()
abstract_state = jax.eval_shape(init_fn, *init_args)
best_state = mngr.restore(best_step, args=ocp.args.StandardRestore(abstract_state))
```

`best_fn` extracts a scalar from the metrics dict, and `best_mode` ('min' or 'max') determines the direction. Orbax keeps all recent checkpoints and tracks the best separately.

## Common mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Forget `wait_until_finished()` | Last checkpoint lost on async save | Always call after training loop |
| Restore without concrete state target | TypeError: StandardRestore needs structure | Initialize state first, pass to restore |
| Save full eqx.Module with non-array leaves | Serialization error or data loss | Use `eqx.filter(model, eqx.is_array)` before save |
| Wrong path on multi-host setup | Race condition, corrupted checkpoint | Point all hosts at shared filesystem (GCS, NFS, Lustre); Orbax coordinates barriers internally |
| Check `latest_step()` without None guard | AttributeError on first run | Always: `if mngr.latest_step() is not None:` |
| Assume old checkpoint format compatible | Restore fails with cryptic error | Validate checkpoint version before restore |
