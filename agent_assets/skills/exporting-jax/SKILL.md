---
name: exporting-jax
description: Use when exporting a JAX function to StableHLO with jax.export — static and symbolic shapes, serialization, export parity, Equinox export, checkify, or auditing an exported graph with jaxlint[hlo]. For IREE, WASM, WebGPU or ONNX deployment targets, see xtrax's export reference.
---

# exporting-jax

Export JAX programs to StableHLO with `jax.export`: the portable, serializable artifact that
deployment toolchains start from.

## When to use

- Ahead-of-time export of a jitted function, for serving or for a later compile step
- Serializing a compiled program to bytes and reloading it in another JAX process
- Checking that an exported artifact matches the original function
- Auditing an exported graph for performance or safety issues (`jaxlint[hlo]`)

**Not covered here: compiling the export for another runtime** (IREE native/WASM/Vulkan,
ONNX, the browser). That belongs to [xtrax](https://github.com/maraxen/xtrax)'s export layer and
its `using-xtrax` skill, `references/export.md`. It covers what each target actually proves
(`EXECUTED` vs `CODEGEN_ONLY`), independent parity oracles, and known gaps. In particular,
**IREE 3.11 has no WebGPU backend, and the SPIR-V it emits for Vulkan is rejected by browsers**
(push constants), so a compiled SPIR-V artifact is not evidence of browser readiness.

## StableHLO export requirements

StableHLO is the portable IR JAX lowers to. For a function to export cleanly:

1. **Static shapes.** All array shapes must be known at export time. For a dynamic
   dimension, use `jax.export.symbolic_shape` (below).
2. **No Python control flow on traced values.** Use `jax.lax.cond`, `jax.lax.scan`,
   `jax.lax.while_loop` etc. A static `for` over a fixed-length Python list is fine.
3. **Known output sizes.** `jnp.unique`, `jnp.nonzero` and size-changing `jnp.where`
   need an explicit `size=` to export.
4. **No Python-side randomness.** Pass a PRNG key as an argument; never call `np.random`
   inside the exported function.

```python
import jax
import jax.numpy as jnp

def model_fn(params, x):  # no Python control flow, static shapes
    return jnp.dot(x, params['w']) + params['b']

# Lower to StableHLO MLIR text
lowered = jax.jit(model_fn).lower(params, jnp.ones((batch, features)))
print(lowered.as_text())  # inspect StableHLO
```

## jax.export: the AOT API

`jax.export` (public since JAX 0.4.30; earlier releases had `jax.experimental.export`) is
the recommended AOT path. It produces an `Exported` artifact that serializes to bytes.

```python
import jax
import jax.numpy as jnp

# Export with concrete shapes
exported = jax.export.export(jax.jit(model_fn))(params, jnp.ones((4, 128)))

# Serialize to bytes (e.g., for disk or network)
serialized = exported.serialize()

# Deserialize and call later, in another JAX process. Calling an Exported still
# runs through JAX; it removes the need for the original Python source, not for JAX.
reloaded = jax.export.deserialize(serialized)
result = reloaded.call(params, x)
```

**Shape polymorphism**: export with a symbolic batch dimension for serving:

```python
import jax
import jax.numpy as jnp

batch, = jax.export.symbolic_shape("batch")  # unpack 1-tuple to get a single dim
x_aval = jax.ShapeDtypeStruct((batch, 128), jnp.float32)
exported = jax.export.export(
    jax.jit(model_fn),
    platforms=["cpu"]
)(params, x_aval)
```

**Accessing the exported artifact:**

```python
# MLIR text (lowered StableHLO)
mlir_text = exported.mlir_module()

# Input and output abstract values
input_avals = exported.in_avals    # [AbstractValue for each arg]
output_avals = exported.out_avals  # [AbstractValue for each result]

# Calling with compatible concrete values
result = exported.call(params, x)
```

### Verifying export parity

Before relying on an exported artifact, check it against the original function:

```python
import jax
import jax.numpy as jnp

expected = jax.jit(model_fn)(params, x)

exported = jax.export.export(jax.jit(model_fn))(params, x)
actual = exported.call(params, x)

assert jnp.allclose(expected, actual, atol=1e-5), "Parity check failed"
```

This compares JAX with JAX on the same backend, so it can only catch export and serialization
problems. A deployment target (IREE, ONNX Runtime) needs a parity check against an
**independent** reference on that target, with a negative control that must fail. See xtrax's
`references/parity.md`.

## Auditing exported kernels with jaxlint

The optional `jaxlint[hlo]` extra provides kernel analysis over exported StableHLO:

```python
from jaxlint.hlo import GraphAuditSensor, SemanticAnalyzer

# Audit a kernel for semantic violations (implicit loops, outfeed, etc.)
violations = GraphAuditSensor.audit_kernel(model_fn, params, sample_x)
print(violations)  # {"success": bool, "violations": [...], "operation_counts": {...}}

# Or analyze an already-exported artifact
exported = jax.export.export(jax.jit(model_fn))(params, sample_x)
violations = GraphAuditSensor.audit_exported(exported)

# Deep inspection: extract operation graph
analyzer = SemanticAnalyzer(exported)
summary = analyzer.get_summary()  # operation counts, unique ops, dtypes
operations = analyzer.get_operations()  # OperationInfo rows with locations
provenance = analyzer.get_type_provenance()  # input/output shapes and dtypes
```

**Note:** `GraphAuditSensor.audit_kernel` wraps `fn` with `checkify.checkify` before export.
The resulting `Exported` artifact has extra error-state outputs. Use `audit_exported` directly
on a pre-exported artifact to inspect the unchecked graph.

## Common export gotchas

| Gotcha | Symptom | Fix |
|--------|---------|-----|
| Dynamic shape in export | `TracerBoolConversionError` or shape mismatch at serialize | Pad to static shape before export; use `symbolic_shape` for dynamic dims |
| Python `if` on traced value | `ConcretizationTypeError` during `lower()` | Replace with `jax.lax.cond` |
| `jnp.unique` / `jnp.nonzero` | "output shapes not known" error | Pass `size=N` (e.g. `jnp.unique(x, size=N)`) or pre-allocate a known max size |
| PRNG inside exported fn | Non-deterministic or error on deserialize | Pass `key` as explicit argument, call `jax.random.split` outside |
| Equinox Module export | Non-array leaves in the exported signature | Call `eqx.partition(model, eqx.is_array)` to separate arrays and statics before export |

## Equinox model export pattern

Equinox modules are JAX pytrees but carry non-array leaves. Separate arrays from statics before
export:

```python
import equinox as eqx
import jax

model = MyModel(...)
params, static = eqx.partition(model, eqx.is_array)

def model_fn(params, x):
    model = eqx.combine(params, static)  # static captured in closure
    return model(x)

# Export params as the first arg; static is a closure constant
exported = jax.export.export(jax.jit(model_fn))(params, sample_x)
mlir_text = exported.mlir_module()  # StableHLO, input to any downstream compiler
```

## Checkify + export for safe kernels

Combine `jax.experimental.checkify` with export to carry safety checks through to the artifact:

```python
from jax.experimental import checkify
import jax
import jax.numpy as jnp

def safe_model_fn(params, x):
    checkify.check(jnp.all(x > 0), "inputs must be positive")
    return jnp.dot(x, params['w']) + params['b']

# checkify threads the check through as explicit error state
checked_fn = checkify.checkify(safe_model_fn, errors=checkify.all_checks)

# The exported function returns (error, result); inspect `error` after each call
exported = jax.export.export(jax.jit(checked_fn))(params, sample_x)
err, out = exported.call(params, x)
err.throw()  # raises if a check failed; err.get() returns the message or None
```

The error state is an extra output of the exported program, not a runtime abort or a
`stablehlo` assert op. Callers must check it; this is the same effect as the `audit_kernel` note above.

**A checkified export cannot be serialized.** Verified on jax 0.11.1:
`exported.serialize()` raises `TypeError: Serialization is supported only for dictionaries with
string keys`, because checkify's `Error` pytree is keyed by `ErrorEffect` objects. To ship it,
return the error's leaves as plain arrays (`jax.tree_util.tree_leaves(err)`: boolean
"check failed" flags and integer codes) and turn them back into errors on the host. The
messages themselves do not survive.
