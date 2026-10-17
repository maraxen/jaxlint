# jaxlint-expect: JD002, JL002, JL003
"""Replicate-style loops that must NOT be flagged by JL021."""

import time

import jax
import jax.numpy as jnp
import numpy as np


def progress_only(n_boot):
    """Replicate loop whose body only prints."""
    for b in range(n_boot):  # noflag
        print(b)


def unrelated_append(n):
    """Accumulating loop over a name that is not a replicate axis."""
    out = []
    for i in range(n):  # noflag
        out.append(float(i) ** 2)
    return out


def no_accumulation(x, n_boot):
    """Replicate loop that calls a function but keeps no per-iteration result."""
    for _ in range(n_boot):  # noflag
        _ = np.linalg.norm(x)


def timing(n_reps, fn):
    """Timing loop whose body only calls time functions for the clock."""
    times = []
    for _ in range(n_reps):  # noflag
        t0 = time.perf_counter()
        times.append(time.perf_counter() - t0)
    return times


@jax.jit
def kernel_loop(n_boot, x):
    """Python loop inside a jit kernel is JL003's concern, not JL021's."""
    acc = x
    for _ in range(n_boot):  # noflag
        acc = acc + jnp.sin(acc)
    return acc


def one_replicate(key, x):
    """Single-replicate kernel body."""
    return jnp.sum(jax.random.normal(key, x.shape) * x)


def vmapped(keys, x):
    """Batched replicate axis: the form JL021 asks for."""
    return jax.vmap(one_replicate, in_axes=(0, None))(keys, x)
