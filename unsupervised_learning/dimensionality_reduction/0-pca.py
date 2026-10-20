#!/usr/bin/env python3
"""PCA that keeps a fraction of the variance"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset

    X: numpy.ndarray of shape (n, d), zero-mean across data points
    var: fraction of the variance to maintain
    Returns: W, weights matrix of shape (d, nd)
    """
    _, S, Vt = np.linalg.svd(X)
    ratio = np.cumsum(S) / np.sum(S)
    nd = np.argmax(ratio >= var) + 1
    return Vt.T[:, :nd]
