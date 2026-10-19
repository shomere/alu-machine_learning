#!/usr/bin/env python3
"""Calculates intra-cluster variance."""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        C (numpy.ndarray): Centroids of shape (k, d).

    Returns:
        float: Total intra-cluster variance.
        None: On failure.
    """
    if (not isinstance(X, np.ndarray) or
            not isinstance(C, np.ndarray) or
            X.ndim != 2 or
            C.ndim != 2 or
            X.shape[1] != C.shape[1]):
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    return np.sum(np.min(distances, axis=1) ** 2)
