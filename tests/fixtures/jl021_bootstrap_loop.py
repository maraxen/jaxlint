# jaxlint-expect: JL021
"""Host-side replicate loops (numpy bootstrap, scipy per-replicate fit) that should be batched."""

import numpy as np
from scipy.optimize import minimize


def bootstrap_mean(data, n_boot):
    """Bootstrap the mean with a host loop and accumulate into a list."""
    rng = np.random.default_rng(0)
    estimates = []
    for _ in range(n_boot):  # flag
        sample = rng.choice(data, size=data.shape[0], replace=True)
        estimates.append(np.mean(sample))
    return np.asarray(estimates)


def nsb_fit_per_replicate(curves, n_boot, x0):
    """One scipy.optimize fit per bootstrap replicate, written into a preallocated array."""
    fits = np.zeros((n_boot, x0.size))
    for b in range(n_boot):  # flag
        resampled = np.random.default_rng(b).choice(curves, size=len(curves))
        res = minimize(lambda p: np.sum((resampled.mean(0) - p) ** 2), x0)
        fits[b] = res.x
    return fits
