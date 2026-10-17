# jaxlint-expect: JM002
"""JM002: malformed \\frac inside display math."""

import jax.numpy as jnp


def ok_doc(a: jnp.ndarray) -> jnp.ndarray:
    r"""Demonstrate malformed LaTeX.

    Bad display math uses ``\\frac`` without braced arguments:
    $$
    \\frac 1 2
    $$

    Parameters
    ----------
    a : jnp.ndarray
        Input.

    Returns
    -------
    jnp.ndarray
        Output.
    """
    return a
