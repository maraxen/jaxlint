# jaxlint-expect: JD002, JL002
"""JL005 false-positive regression: .shape passed to shape-consuming ops should NOT be flagged."""

import jax
import jax.numpy as jnp


@jax.jit
def add_noise(key, coordinates):
    """Canonical same-shape sampling idiom — shape remains polymorphic."""
    noise = jax.random.normal(key, coordinates.shape, dtype=coordinates.dtype)
    return coordinates + noise


@jax.jit
def make_zeros_like_shape(x):
    """jnp.zeros(x.shape) is a shape-propagating op, not a branch on shape."""
    return jnp.zeros(x.shape, dtype=x.dtype)
