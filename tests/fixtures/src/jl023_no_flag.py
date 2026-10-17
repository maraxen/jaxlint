# jaxlint-expect: none
"""JL023 negatives: host syncs that must not be flagged."""

import time

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def step(x):
    return x * 2.0


def sync_after_loop(xs):
    x = jnp.asarray(xs)
    rows = []
    for _ in range(10):
        x = x + 1.0
        rows.append(x)
    stacked = jnp.stack(rows)
    return np.asarray(stacked), float(stacked[0, 0])


def numpy_only_value(n):
    a = np.arange(n)
    total = 0.0
    for i in range(n):
        total += float(a[i])
        total += a[i].item()
    return total


def timing_loop(xs, reps):
    x = jnp.asarray(xs)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = step(x)
        y.block_until_ready()
        times.append(time.perf_counter() - t0)
    return times


def untainted_name(n):
    count = 0
    for i in range(n):
        k = i * 2
        count += int(k)
    return count
