#!/usr/bin/env python3
"""PCA to a fixed number of dimensions"""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset

    X: numpy.ndarray of shape (n, d)
    ndim: new dimensionality of the transformed X
    Returns: T, numpy.ndarray of shape (n, ndim)
    """
    X_m = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_m, full_matrices=False)
    return np.matmul(X_m, Vt.T[:, :ndim])
