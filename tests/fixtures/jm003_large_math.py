# jaxlint-expect: JM003
"""JM003: display math block large relative to surrounding prose."""

import jax.numpy as jnp


def jm003_public(a: int) -> int:
    """X

    Parameters
    ----------
    a : int
        b

    Returns
    -------
    int
        c

    $$
    \\alpha + \\beta + \\gamma + \\delta + \\epsilon + \\zeta +
    \\eta + \\theta + \\iota + \\kappa + \\lambda + \\mu + \\nu +
    \\xi + \\omicron + \\pi + \\rho + \\sigma + \\tau + \\upsilon +
    \\phi + \\chi + \\psi + \\omega + \\alpha + \\beta + \\gamma + \\delta + \\epsilon + \\zeta +
    \\eta + \\theta + \\iota + \\kappa + \\lambda + \\mu + \\nu +
    \\xi + \\omicron + \\pi + \\rho + \\sigma + \\tau + \\upsilon +
    \\phi + \\chi + \\psi + \\omega
    $$
    """
    _ = jnp.asarray(a)
    return a + 1
