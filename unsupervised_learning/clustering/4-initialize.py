#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        pi, m, S or (None, None, None) on failure.
    """

    if (not isinstance(X, np.ndarray) or
            X.ndim != 2):
        return None, None, None

    n, d = X.shape

    if (not isinstance(k, int) or
            k <= 0 or
            k > n):
        return None, None, None

    # Equal priors
    pi = np.full((k,), 1 / k)

    # Means from K-means
    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    # Identity covariance matrices
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
