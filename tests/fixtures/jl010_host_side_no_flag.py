"""JL010 false-positive regression: host-side f64 casts should NOT be flagged."""

import jax.numpy as jnp


def grid_spacing(box_size, grid_points):
    """f64 cast immediately consumed by float() — never enters a kernel."""
    return float(jnp.mean(box_size.astype(jnp.float64))) / float(max(grid_points, 1))


def box_mean(box_arr):
    """f64 immediately cast to Python float — never enters a kernel."""
    return float(jnp.mean(box_arr.astype(jnp.float64)))


def template_zeros(n_atoms):
    """f64 as dtype= kwarg — template/canonical usage, not kernel arithmetic."""
    return jnp.zeros((n_atoms, 3), dtype=jnp.float64)
