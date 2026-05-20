---
name: exporting-jax
description: Use when exporting a JAX model to StableHLO, compiling with IREE for WASM/native targets, or using jax.export for ahead-of-time compilation.
---

# exporting-jax

Export JAX programs to portable intermediate representations for serving, embedding, or cross-platform deployment.

## When to use

- Deploying a JAX model to a non-JAX runtime (browser, mobile, embedded)
- Compiling ahead-of-time for latency-critical serving
- Interoperating with TensorFlow SavedModel or ONNX ecosystems
- Running inference in WASM (browser/server-side without GPU)
- Auditing kernel graphs for performance or safety violations
- Capturing model state without JAX dependency in deployment

## StableHLO export requirements

StableHLO is the portable IR format that JAX compiles to. For a function to export cleanly:

1. **Static shapes** — all array shapes must be known at export time. No dynamic shapes from Python. Use `jax.export.symbolic_shape` only if you need dynamic dims (advanced).
2. **No dynamic control flow on traced values** — dynamic branches and iteration counts must not depend on traced array values. Use `jax.lax.cond`, `jax.lax.scan`, etc. for conditional or looping logic over arrays. (Static `for` over a fixed-length Python list is fine.)
3. **Supported ops** — most `jnp.*` ops are supported. Watch for: `jnp.unique`, `jnp.nonzero`, `jnp.where` with dynamic shapes — these require known output sizes.
4. **No Python-side randomness** — pass PRNGKey as an explicit argument; do not use `np.random` inside the exported function.

```python
import jax
import jax.numpy as jnp

def model_fn(params, x):  # no Python control flow, static shapes
    return jnp.dot(x, params['w']) + params['b']

# Lower to StableHLO MLIR text
lowered = jax.jit(model_fn).lower(params, jnp.ones((batch, features)))
print(lowered.as_text())  # inspect StableHLO
```

## jax.export — native AOT API

`jax.export` (added in JAX 0.4.14+) is the recommended AOT path. It produces a portable artifact that can be serialized and deserialized without JAX.

```python
import jax
import jax.export

# Export with concrete shapes
exported = jax.export.export(jax.jit(model_fn))(params, jnp.ones((4, 128)))

# Serialize to bytes (e.g., for disk or network)
serialized = exported.serialize()

# Deserialize and call in another process (no JAX needed for execution)
reloaded = jax.export.deserialize(serialized)
result = reloaded.call(params, x)
```

**Shape polymorphism** — export with symbolic batch dimension for serving:

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

Before deploying an exported artifact, verify that it produces identical results to the original JAX function:

```python
import jax
import jax.numpy as jnp

# Run in JAX:
expected = jax.jit(model_fn)(params, x)

# Run via exported:
exported = jax.export.export(jax.jit(model_fn))(params, x)
actual = exported.call(params, x)

# Compare:
assert jnp.allclose(expected, actual, atol=1e-5), f"Parity check failed"
# Note: small numerical differences (1e-6 range) are normal due to fusion ordering
```

## IREE compilation pipeline

IREE compiles StableHLO to native code for CPU, GPU, WASM, and Vulkan. The typical path from JAX:

```python
# Step 1: Export and get StableHLO MLIR text from JAX
exported = jax.export.export(jax.jit(model_fn))(params, sample_input)
mlir_text = exported.mlir_module()  # guaranteed StableHLO MLIR string

# Save to file
with open("model.mlir", "w") as f:
    f.write(mlir_text)

# Step 2: compile with iree-compile (CLI)
# iree-compile --iree-input-type=stablehlo \
#              --iree-hal-target-backends=llvm-cpu \
#              --output-format=vmfb \
#              model.mlir -o model.vmfb

# Step 3: run with iree-run-module or Python bindings
import iree.runtime as ireert

config = ireert.Config("local-task")
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.mmap(ctx.instance, "model.vmfb")
ctx.add_vm_module(vm_module)
runner = ctx.modules.module["main"]
result = runner(iree_params, iree_input)
```

**CPU / WASM (Emscripten) target** — for browser or Node.js execution without GPU:

```bash
iree-compile --iree-input-type=stablehlo \
             --iree-hal-target-backends=llvm-cpu \
             --output-format=vmfb \
             model.mlir -o model_cpu.vmfb

# For WASM specifically, add Emscripten flags:
# --iree-llvmcpu-target-triple=wasm32-unknown-emscripten
# --iree-llvmcpu-target-cpu-features=+atomics,+bulk-memory
```

**WebGPU target** — for GPU in the browser (NOT WASM):

```bash
iree-compile --iree-input-type=stablehlo \
             --iree-hal-target-backends=webgpu-spirv \
             --output-format=vmfb \
             model.mlir -o model_webgpu.vmfb
```

**Key requirements for IREE:**
- All shapes static at compile time (or use dynamic shape API)
- No custom Python callbacks inside the compiled function
- `iree-compile` version must match `iree-runtime` version (pin both in `pyproject.toml`)
- Output file is a VMFB (IREE VM FlatBuffer); use `--output-format=mlir` for debugging

## WebGPU / WASM deployment

JAX does not currently export directly to WASM via `jax.export`. Valid platforms are `"cpu"`, `"cuda"`, `"tpu"`, `"rocm"`. For WASM and WebGPU deployment:

- **IREE + Emscripten**: compile with `--iree-hal-target-backends=llvm-cpu` and `--iree-llvmcpu-target-triple=wasm32-unknown-emscripten` for WASM bytecode.
- **ONNX Runtime Web**: convert via `jax2onnx` then load in ONNX Runtime Web.

For GPU in the browser, use the IREE WebGPU target (see IREE compilation pipeline section above).

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

**Note:** `GraphAuditSensor.audit_kernel` wraps `fn` with `checkify.checkify` before export. The resulting `Exported` artifact has extra error-state outputs. Use `audit_exported` directly on a pre-exported artifact to inspect the unchecked graph.

## Common export gotchas

| Gotcha | Symptom | Fix |
|--------|---------|-----|
| Dynamic shape in export | `TracerBoolConversionError` or shape mismatch at serialize | Pad to static shape before export; use `symbolic_shape` for dynamic dims |
| Python `if` on traced value | `ConcretizationTypeError` during `lower()` | Replace with `jax.lax.cond` |
| `jnp.unique` / `jnp.nonzero` | "output shapes not known" error | Pre-allocate output buffers with known max size or use `jnp.unique(..., size=N)` |
| PRNG inside exported fn | Non-deterministic or error on deserialize | Pass `key` as explicit argument, call `jax.random.split` outside |
| IREE version mismatch | Segfault or "invalid VMFB" on load | Pin `iree-compiler` and `iree-runtime` to exact same version |
| Equinox Module export | `eqx.Module` is not a plain pytree for IREE | Call `eqx.partition(model, eqx.is_array)` to separate arrays and statics before export |

## Equinox model export pattern

Equinox modules are JAX pytrees but carry non-array leaves. Separate arrays from statics before export:

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

# IREE compilation
mlir_text = exported.mlir_module()
# ... iree-compile model.mlir -o model.vmfb
```

## Checkify + export for safe kernels

Combine `jax.experimental.checkify` with export to capture safety checks in StableHLO:

```python
from jax.experimental import checkify
import jax
import jax.numpy as jnp

def safe_model_fn(params, x):
    checkify.check(jnp.all(x > 0), "inputs must be positive")
    return jnp.dot(x, params['w']) + params['b']

# Wrap with checkify to embed checks as StableHLO ops
checked_fn = checkify.checkify(safe_model_fn, errors=checkify.all_checks)

# Export the checked function (error state is an extra output)
exported = jax.export.export(jax.jit(checked_fn))(params, sample_x)
# Note: exported has an extra error-state output — see audit_kernel note below
```

Checks embed as `stablehlo.assert` and `stablehlo.custom_call` ops in MLIR, visible in audits.
