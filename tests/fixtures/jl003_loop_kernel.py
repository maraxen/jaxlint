"""JL003: Python loop inside JAX kernel."""

import jax


@jax.jit
def _loops(x):
    s = 0
    for _ in range(3):
        s = s + x
    return s
