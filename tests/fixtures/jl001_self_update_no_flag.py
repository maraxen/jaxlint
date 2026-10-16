"""JL001 false-positive regression: self-update scatter should NOT be flagged.

Buffer-fill patterns like Q = Q.at[:, step].set(v) appear in lax.scan bodies
where later reads access disjoint indices.
"""

import jax
import jax.numpy as jnp


@jax.jit
def lanczos_basis(A, q0, n_steps):
    """Fills a Lanczos basis Q column by column; reads disjoint columns later."""
    Q = jnp.zeros((q0.shape[0], n_steps))
    Q = Q.at[:, 0].set(q0)  # self-update: lhs == rhs base
    beta_arr = jnp.zeros(n_steps)
    beta_arr = beta_arr.at[0].set(0.0)  # self-update
    # later read is a different column — no scatter+gather anti-pattern
    _ = Q[:, 0] + beta_arr[0]
    return Q


@jax.jit
def pme_weights(w):
    """Chained scatter: second .at[].set() is not a gather of the first."""
    w = w.at[:, :, 0].set(1.0)  # self-update
    w = w.at[:, :, -1].set(1.0)  # second scatter, not a gather
    return w
