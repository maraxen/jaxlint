"""JL003/JL004/JL005 under an Equinox filter_jit decorator (alias form)."""

import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def _eqx_loops(x):
    s = 0
    for _ in range(3):  # JL003 trigger
        s = s + x
    return s


@eqx.filter_jit
def _eqx_print(x):
    print("side effect")  # JL004 trigger
    return x + 1


@eqx.filter_jit
def _eqx_shape(x):
    _ = x.shape  # JL005 trigger
    return jnp.sum(x)
