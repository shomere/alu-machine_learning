#!/usr/bin/env python3
"""Expectation step for a GMM."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d)
        pi: numpy.ndarray of shape (k,)
        m: numpy.ndarray of shape (k, d)
        S: numpy.ndarray of shape (k, d, d)

    Returns:
        g, l
        or (None, None) on failure.
    """

    if (not isinstance(X, np.ndarray) or X.ndim != 2):
        return None, None

    if (not isinstance(pi, np.ndarray) or pi.ndim != 1):
        return None, None

    if (not isinstance(m, np.ndarray) or m.ndim != 2):
        return None, None

    if (not isinstance(S, np.ndarray) or S.ndim != 3):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (m.shape != (k, d) or
            S.shape != (k, d, d)):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])

        if P is None:
            return None, None

        g[i] = pi[i] * P

    total = np.sum(g, axis=0)

    g /= total

    l = np.sum(np.log(total))

    return g, l
