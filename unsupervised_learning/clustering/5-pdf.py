#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): shape (n, d)
        m (numpy.ndarray): shape (d,)
        S (numpy.ndarray): shape (d, d)

    Returns:
        P (numpy.ndarray): shape (n,)
        or None on failure.
    """

    if (not isinstance(X, np.ndarray) or
            not isinstance(m, np.ndarray) or
            not isinstance(S, np.ndarray)):
        return None

    if X.ndim != 2 or m.ndim != 1 or S.ndim != 2:
        return None

    n, d = X.shape

    if m.shape != (d,) or S.shape != (d, d):
        return None

    det = np.linalg.det(S)

    if det <= 0:
        return None

    inv = np.linalg.inv(S)

    diff = X - m

    exponent = np.sum((diff @ inv) * diff, axis=1)

    norm = np.sqrt(((2 * np.pi) ** d) * det)

    P = np.exp(-0.5 * exponent) / norm

    return np.maximum(P, 1e-300)
