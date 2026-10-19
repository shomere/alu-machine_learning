#!/usr/bin/env python3
"""K-means clustering"""

initialize = __import__("0-initialize").initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: number of clusters
        iterations: maximum iterations

    Returns:
        C, clss
        or (None, None) on failure
    """

    if (not isinstance(X, np.ndarray) or
            X.ndim != 2 or
            not isinstance(k, int) or
            k <= 0 or
            k > X.shape[0] or
            not isinstance(iterations, int) or
            iterations <= 0):
        return None, None

    n, d = X.shape

    C = initialize(X, k)

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    for _ in range(iterations):

        old_C = C.copy()

        # Assign each point to the nearest centroid
        distances = np.linalg.norm(
            X[:, np.newaxis] - C,
            axis=2
        )

        clss = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):

            points = X[clss == i]

            if points.shape[0] == 0:
                C[i] = np.random.uniform(mins, maxs)
            else:
                C[i] = np.mean(points, axis=0)

        if np.array_equal(C, old_C):
            break

    return C, clss
