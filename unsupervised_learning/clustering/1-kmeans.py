#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: positive integer containing the number of clusters
        iterations: maximum number of iterations

    Returns:
        C, clss
            C is a numpy.ndarray of shape (k, d) containing
            the centroid means
            clss is a numpy.ndarray of shape (n,) containing
            the index of the cluster for each data point

        Returns (None, None) on failure.
    """

    if (not isinstance(X, np.ndarray) or
            len(X.shape) != 2):
        return None, None

    if (not isinstance(k, int) or
            k <= 0 or
            k > X.shape[0]):
        return None, None

    if (not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    n, d = X.shape

    # Initialize centroids (1st call to np.random.uniform)
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    C = np.random.uniform(low=low, high=high, size=(k, d))

    for _ in range(iterations):

        old_C = C.copy()

        # Compute distances
        distances = np.linalg.norm(
            X[:, np.newaxis] - C,
            axis=2
        )

        # Assign clusters
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):

            points = X[clss == i]

            if points.shape[0] == 0:
                # Reinitialize empty cluster
                C[i] = np.random.uniform(low=low, high=high)
            else:
                C[i] = np.mean(points, axis=0)

        # Early stopping
        if np.array_equal(old_C, C):
            return C, clss

    return C, clss
