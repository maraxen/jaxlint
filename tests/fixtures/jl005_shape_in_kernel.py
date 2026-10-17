# jaxlint-expect: JL002, JL005
"""JL005: .shape introspection inside kernel."""

import jax
import jax.numpy as jnp


@jax.jit
def _needs_shape(x):
    _ = x.shape  # JL005 trigger
    return jnp.sum(x)
