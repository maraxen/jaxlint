# jaxlint-expect: JL022
"""JL022: jit-family wrap of a fresh callable inside a host loop (retraces every iteration)."""

import functools

import equinox as eqx
import jax

SCALE = 2.0


def scale(x, k):
    return x * k


def run_lambda(ys):
    out = []
    for _ in range(3):
        f = jax.jit(lambda x: x * 2)
        out.append(f(ys))
    return out


def run_partial(ys):
    out = []
    for _ in range(3):
        g = jax.jit(functools.partial(scale, k=SCALE))
        out.append(g(ys))
    return out


def run_nested_def(ys):
    out = []
    for _ in range(3):

        def step(x):
            return x + 1.0

        f = jax.jit(step)
        out.append(f(ys))
    return out


def run_decorated_def(ys):
    out = []
    i = 0
    while i < 3:

        @jax.jit
        def inner(x):
            return x * SCALE

        out.append(inner(ys))
        i += 1
    return out


def run_filter_jit(ys):
    out = []
    for _ in range(3):
        h = eqx.filter_jit(lambda x: x - 1.0)
        out.append(h(ys))
    return out
