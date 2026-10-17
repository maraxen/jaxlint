# jaxlint-expect: JD001
"""JL023 default scope: the same positive pattern outside a src/ path must not fire."""

import jax.numpy as jnp


def item_in_loop(xs):
    x = jnp.asarray(xs)
    total = 0.0
    for _ in range(10):
        x = x + 1.0
        total += x.item()
    return total
