# jaxlint-expect: JL023
"""JL023 positives: device values forced to host inside host-side loops."""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def step(x):
    return x * 2.0


def item_in_loop(xs):
    x = jnp.asarray(xs)
    total = 0.0
    for _ in range(10):
        x = x + 1.0
        total += x.item()  # sync-here
    return total


def tolist_and_casts(xs):
    y = jnp.sqrt(jnp.asarray(xs))
    out = []
    for i in range(4):
        out.append(y.tolist())  # sync-here
        out.append(float(y))  # sync-here
        out.append(int(y[i]))  # sync-here
        out.append(bool(y[0]))  # sync-here
    return out


def numpy_and_device_get(xs):
    z = jnp.asarray(xs)
    acc = []
    i = 0
    while i < 3:
        acc.append(np.asarray(z))  # sync-here
        acc.append(np.array(z))  # sync-here
        acc.append(jax.device_get(z))  # sync-here
        i += 1
    return acc


def block_in_loop(xs):
    state = jnp.asarray(xs)
    for _ in range(5):
        state = step(state)
        state.block_until_ready()  # sync-here
        jax.block_until_ready(state)  # sync-here
    return state


def propagated_and_unpacked(xs):
    a = jnp.asarray(xs)
    b = a  # simple assignment propagates taint
    lo, hi = step(a), step(b)
    results = []
    for _ in range(2):
        results.append(float(b))  # sync-here
        results.append(hi.item())  # sync-here
        results.append(lo[0].item())  # sync-here
    return results
