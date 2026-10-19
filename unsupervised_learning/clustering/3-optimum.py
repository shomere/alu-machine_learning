#!/usr/bin/env python3
"""Finds the optimum number of clusters for K-means."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d)
        kmin: minimum number of clusters
        kmax: maximum number of clusters
        iterations: maximum iterations for K-means

    Returns:
        results, d_vars
        or (None, None) on failure.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n = X.shape[0]

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or
            kmax <= kmin or
            kmax > n):
        return None, None

    if (not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    results = []
    vars_list = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None:
            return None, None

        results.append((C, clss))
        vars_list.append(variance(X, C))

    d_vars = [vars_list[0] - v for v in vars_list]

    return results, d_vars
