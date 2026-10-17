# jaxlint-expect: none
import time

import jax
import jax.numpy as jnp
from otherpkg.kernels import step_jit


@jax.jit
def step(x):
    return x * 0.9 + 1.0


@jax.jit
def apply_j(state):
    return state * 0.5


@jax.jit
def pair_step(x, k):
    return x + 1.0, k + 1


def carry_recurrence(x0, n):
    x = x0
    out = []
    for _ in range(n):
        x = step(x)
        out.append(x)
    return out


def serial_simulation(state, steps):
    energies = []
    for _ in range(steps):
        state = apply_j(state)
        energies.append(float(jnp.sum(state)))
    return energies


def serial_pair(x, k, n):
    out = []
    for _ in range(n):
        x, k = pair_step(x, k)
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


def helper_timed_loop(xs):
    results = []
    for x in xs:
        results.append(bench_once(step, x))
    return results


def bench_once(fn, x):
    return fn(x)


def imported_jit_loop(rows):
    out = []
    for row in rows:
        out.append(step_jit(row))
    return out


def jnp_default_off(rows):
    out = []
    for row in rows:
        out.append(jnp.sum(row))
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
        out.append(step(x))
    return out
