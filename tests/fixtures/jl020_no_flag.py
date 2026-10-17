# jaxlint-expect: none
import time

import jax
import jax.numpy as jnp
from otherpkg.kernels import step_jit


@jax.jit
def step(x):
    return x * 0.9 + 1.0


def carry_recurrence(x0, n):
    x = x0
    out = []
    for _ in range(n):
        x = step(x)
        out.append(x)
    return out


def timed_loop(xs):
    results = []
    times = []
    for x in xs:
        t0 = time.perf_counter()
        results.append(step(x))
        times.append(time.perf_counter() - t0)
    return results, times


def imported_jit_loop(rows):
    out = []
    for row in rows:
        out.append(step_jit(row))
    return out


def vmapped(rows):
    return jax.vmap(step)(rows)


def unused_result(xs):
    for x in xs:
        step(x)


@jax.jit
def kernel_loop(xs):
    out = []
    for x in xs:
        out.append(jnp.sum(x))
    return out
