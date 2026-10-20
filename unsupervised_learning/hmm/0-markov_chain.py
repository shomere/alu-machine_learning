#!/usr/bin/env python3
"""Markov chain state probabilities"""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chain being in each state
    after t iterations

    P: square numpy.ndarray of shape (n, n), the transition matrix
    s: numpy.ndarray of shape (1, n), the starting probabilities
    t: number of iterations
    Returns: numpy.ndarray of shape (1, n), or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, P.shape[0]):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    return np.matmul(s, np.linalg.matrix_power(P, t))
