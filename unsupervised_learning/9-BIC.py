#!/usr/bin/env python3
"""Bayesian Information Criterion (BIC) for Gaussian Mixture Models."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive int, minimum number of clusters to check (inclusive).
        kmax: positive int, maximum number of clusters to check (inclusive).
            If None, defaults to n.
        iterations: positive int, max iterations for the EM algorithm.
        tol: non-negative float, tolerance for the EM algorithm.
        verbose: bool, whether EM prints information to stdout.

    Returns:
        tuple: (best_k, best_result, l, b) on success, or
            (None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape
    if n == 0 or d == 0:
        return None, None, None, None

    if not isinstance(kmin, int) or isinstance(kmin, bool) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        if kmin > n:
            return None, None, None, None
        kmax = n

    if (not isinstance(kmax, int) or isinstance(kmax, bool) or
            kmax < 1 or kmax < kmin or kmax > n):
        return None, None, None, None

    if (not isinstance(iterations, int) or isinstance(iterations, bool) or
            iterations < 1):
        return None, None, None, None

    if (not isinstance(tol, (int, float)) or isinstance(tol, bool) or
            tol < 0):
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    num_models = kmax - kmin + 1
    l = np.zeros(num_models, dtype=np.float64)
    b = np.zeros(num_models, dtype=np.float64)
    results = []

    cov_params = d * (d + 1) // 2
    log_n = np.log(n)

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        if pi is None or m is None or S is None or log_likelihood is None:
            return None, None, None, None

        p = k * cov_params + k * d + (k - 1)
        bic_val = p * log_n - 2 * log_likelihood

        l[idx] = log_likelihood
        b[idx] = bic_val
        results.append((pi, m, S))

    best_idx = int(np.argmin(b))
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
