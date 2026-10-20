#!/usr/bin/env python3
"""Maximization step for a GMM."""

import numpy as np


def maximization(X, g):
    """
    Performs the maximization step in the EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d)
        g: numpy.ndarray of shape (k, n)

    Returns:
        pi, m, S
        or (None, None, None) on failure.
    """

    if (not isinstance(X, np.ndarray) or
            X.ndim != 2):
        return None, None, None

    if (not isinstance(g, np.ndarray) or
            g.ndim != 2):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    # Effective number of points assigned to each cluster
    Nk = np.sum(g, axis=1)

    # Updated priors
    pi = Nk / n

    # Updated means
    m = (g @ X) / Nk[:, np.newaxis]

    # Updated covariance matrices
    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]
        S[i] = ((g[i][:, np.newaxis] * diff).T @ diff) / Nk[i]

    return pi, m, S
