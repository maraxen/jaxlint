"""JL012: division inside @jax.jit kernel."""

import jax


@jax.jit
def _div(x, y):
    return x / y
