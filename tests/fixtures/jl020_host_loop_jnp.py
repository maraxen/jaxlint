# jaxlint-expect: JL020
# jaxlint-config: host-loop-jnp-ops=true
import jax.numpy as jnp


def run_jnp_loop(rows):
    out = []
    for row in rows:  # hit
        out.append(jnp.sum(row))
    return out
