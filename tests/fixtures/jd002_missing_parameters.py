# jaxlint-expect: JD002
"""JD002: docstring lacks required Parameters."""

import jax.numpy as jnp


def has_returns_missing_params(a: jnp.ndarray) -> jnp.ndarray:
    """Return the input unchanged.

    Returns
    -------
    jnp.ndarray
        Same tensor.
    """
    return a
