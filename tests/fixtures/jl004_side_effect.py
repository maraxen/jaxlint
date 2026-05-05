"""JL004: print inside JAX kernel."""

import jax


@jax.jit
def _dirty(x):
    print("side effect")
    return x + 1
