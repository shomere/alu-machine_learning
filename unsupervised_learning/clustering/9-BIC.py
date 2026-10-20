#!/usr/bin/env python3
"""Bayesian Information Criterion"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None,
        iterations=1000,
        tol=1e-5,
        verbose=False):
    """
    Finds the best number of clusters using BIC.

    Args:
        X: numpy.ndarray of shape (n, d)
        kmin: minimum number of clusters
        kmax: maximum number of clusters
        iterations: maximum EM iterations
        tol: tolerance
        verbose: print EM progress

    Returns:
        best_k, best_result, l, b
        or (None, None, None, None) on failure.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if (not isinstance(kmax, int) or
            kmax <= 0 or
            kmax < kmin or
            kmax > n):
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    num_models = kmax - kmin + 1

    log_likelihoods = np.zeros(num_models)
    bics = np.zeros(num_models)

    best_bic = np.inf
    best_k = None
    best_result = None

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X,
            k,
            iterations,
            tol,
            verbose
        )

        if pi is None:
            return None, None, None, None

        log_likelihoods[idx] = log_likelihood

        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        bics[idx] = p * np.log(n) - 2 * log_likelihood

        if bics[idx] < best_bic:
            best_bic = bics[idx]
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, log_likelihoods, bics
