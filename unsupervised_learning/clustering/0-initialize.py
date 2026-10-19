#!/usr/bin/env python3
"""Initialize K-means centroids"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
        X (numpy.ndarray): dataset of shape (n, d)
        k (int): number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids,
        or None on failure.
    """
    if (not isinstance(X, np.ndarray) or
            X.ndim != 2 or
            not isinstance(k, int) or
            k <= 0 or
            k > X.shape[0]):
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    return np.random.uniform(low=mins, high=maxs,
                             size=(k, X.shape[1]))
