# jaxlint-expect: JL002
"""JL002: short @jax.jit helper."""

import jax


@jax.jit
def _short_helper(x):
    """Keep body tiny (<15 stmt nodes in heuristic)."""
    return x + 1
