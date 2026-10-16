"""JL012 false-positive regression: provably non-zero divisors should NOT be flagged."""

import jax
import jax.numpy as jnp


@jax.jit
def masked_mean(losses, mask):
    """jnp.maximum floor prevents zero divisor."""
    return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@jax.jit
def safe_step(displacement, force):
    """eps addition prevents zero divisor."""
    max_force = jnp.max(jnp.abs(force)) + 1e-8
    return displacement / max_force
