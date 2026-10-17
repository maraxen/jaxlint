# jaxlint-expect: JL020
import equinox as eqx
import jax


@jax.jit
def scale(row):
    return row * 2.0


@eqx.filter_jit
def fit(row):
    return row.sum()


def run_jit_loop(rows):
    out = []
    for row in rows:  # hit
        out.append(scale(row))
    return out


def run_subscript(rows, n):
    arr = [0.0] * n
    for i in range(n):  # hit
        arr[i] = fit(rows[i])
    return arr
