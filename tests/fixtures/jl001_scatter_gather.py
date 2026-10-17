# jaxlint-expect: JD002, JL001, JL002
import jax


@jax.jit
def bad(h, idx, neighbor_idx):
    """Violates JL001."""
    x = h.at[idx].set(1.0)
    return x[neighbor_idx]
