#!/usr/bin/env python3
"""Expectation step for a Gaussian Mixture Model"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        pi: numpy.ndarray of shape (k,) containing the priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices

    Returns:
        g, l
            g: numpy.ndarray of shape (k, n) containing the posterior
               probabilities for each data point in each cluster
            l: total log likelihood

        Returns (None, None) on failure.
    """

    # Validate X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # Validate pi
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    # Validate m
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    # Validate S
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Validate shapes
    if m.shape != (k, d):
        return None, None

    if S.shape != (k, d, d):
        return None, None

    # Validate priors
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Posterior probabilities
    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])

        if P is None:
            return None, None

        g[i] = pi[i] * P

    # Total probability for each data point
    total = np.sum(g, axis=0)

    if np.any(total <= 0):
        return None, None

    # Normalize responsibilities
    g /= total

    # Log likelihood
    l = np.sum(np.log(total))

    return g, l
