# jaxlint-expect: JD001, JL020
"""JL022 negatives: wraps that do not retrace per iteration, or whose callable varies per iteration."""

import jax


def step(x):
    return x + 1.0


def same_function_rewrapped(ys):
    out = []
    for _ in range(3):
        f = jax.jit(step)  # same function object every iteration: cache hit
        out.append(f(ys))
    return out


def lambda_closes_over_loop_var(ys):
    out = []
    for i in range(3):
        f = jax.jit(lambda x: x + i)  # depends on the loop target: genuinely differs
        out.append(f(ys))
    return out


def closure_over_reassigned_name(ys):
    out = []
    for i in range(3):
        k = i * 2.0  # reassigned inside the loop body
        f = jax.jit(lambda x: x * k)
        out.append(f(ys))
    return out


def jit_outside_loop(ys):
    f = jax.jit(lambda x: x * 2)  # wrapped once, before the loop
    out = []
    for _ in range(3):
        out.append(f(ys))
    return out


def default_arg_binds_loop_value(ys):
    out = []
    for i in range(3):
        f = jax.jit(lambda x, k=i: x * k)  # default argument captures the loop value
        out.append(f(ys))
    return out
