#!/usr/bin/env python3
"""Steady state of a regular markov chain"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain

    P: square numpy.ndarray of shape (n, n), the transition matrix
    Returns: numpy.ndarray of shape (1, n), or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n or n == 0:
        return None
    if np.any(P < 0) or not np.allclose(np.sum(P, axis=1), 1):
        return None
    # a chain is regular if some power of P has only positive entries
    # (Wielandt's bound: (n - 1) ** 2 + 1 steps are enough)
    if np.any(np.linalg.matrix_power(P, (n - 1) ** 2 + 1) <= 0):
        return None
    # solve s P = s with the entries of s summing to 1
    A = P.T - np.eye(n)
    A[-1] = 1
    b = np.zeros(n)
    b[-1] = 1
    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return s.reshape(1, n)
