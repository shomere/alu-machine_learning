#!/usr/bin/env python3
"""Absorbing markov chains"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing

    P: square numpy.ndarray of shape (n, n), the transition matrix
    Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape[1] != n or n == 0:
        return False
    if np.any(P < 0) or not np.allclose(np.sum(P, axis=1), 1):
        return False
    absorbing_states = np.diagonal(P) == 1
    if not np.any(absorbing_states):
        return False
    # reach[i, j] > 0 when state j can be reached from state i
    step = (P > 0).astype(float) + np.eye(n)
    reach = np.linalg.matrix_power(step, n) > 0
    # every state must be able to reach at least one absorbing state
    return bool(np.all(np.any(reach[:, absorbing_states], axis=1)))
