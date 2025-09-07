import numpy as np


def hopkins(X: np.ndarray):
    """https://en.wikipedia.org/wiki/Hopkins_statistic"""
    sample_size = 500
    X_tilde = X[np.random.choice(X.shape[0], sample_size, replace=False)]

    # Compute random uniform sample from bounding box
    maxes = X.max(axis=0)
    mins = X.min(axis=0)
    rands = np.random.random(size=(sample_size, X.shape[-1]))
    Y: np.ndarray = (rands * (maxes - mins)) + mins

    # Compute ∑u_i
    sum_u = 0
    for y in Y:
        sum_u += np.min(np.linalg.norm(X - y, axis=-1))

    # Compute ∑w_i
    sum_w = 0
    for i in range(500):
        dists = np.linalg.norm(X - X_tilde[i], axis=-1)
        dists = dists[dists > 0]
        sum_w += np.min(dists)

    return sum_u / (sum_u + sum_w)
