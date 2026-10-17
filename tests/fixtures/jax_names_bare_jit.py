"""JL003/JL004/JL005 under a bare @jit imported from jax (from-import form)."""

import jax.numpy as jnp
from jax import jit


@jit
def _bare_loops(x):
    s = 0
    for _ in range(3):  # JL003 trigger
        s = s + x
    return s


@jit
def _bare_print(x):
    print("side effect")  # JL004 trigger
    return x + 1


@jit
def _bare_shape(x):
    _ = x.shape  # JL005 trigger
    return jnp.sum(x)
