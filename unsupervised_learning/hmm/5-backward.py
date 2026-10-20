#!/usr/bin/env python3
"""The backward algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden markov model

    Observation: numpy.ndarray of shape (T,), observation indices
    Emission: numpy.ndarray of shape (N, M), emission probabilities
    Transition: numpy.ndarray of shape (N, N), transition probabilities
    Initial: numpy.ndarray of shape (N, 1), starting probabilities
    Returns: P, B or None, None on failure
        P: likelihood of the observations given the model
        B: numpy.ndarray of shape (N, T), backward path probabilities
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None
    if np.any(Observation < 0) or np.any(Observation >= M):
        return None, None

    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(Transition * B[:, t + 1] *
                         Emission[:, Observation[t + 1]], axis=1)
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
