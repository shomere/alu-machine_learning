#!/usr/bin/env python3
"""The Viterbi algorithm for a hidden markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states

    Observation: numpy.ndarray of shape (T,), observation indices
    Emission: numpy.ndarray of shape (N, M), emission probabilities
    Transition: numpy.ndarray of shape (N, N), transition probabilities
    Initial: numpy.ndarray of shape (N, 1), starting probabilities
    Returns: path, P or None, None on failure
        path: list of length T with the most likely hidden states
        P: probability of obtaining the path sequence
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

    V = np.zeros((N, T))
    back = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        # scores[i, j]: best path ending in i at t - 1, then moving to j
        scores = V[:, t - 1, np.newaxis] * Transition
        back[:, t] = np.argmax(scores, axis=0)
        V[:, t] = np.max(scores, axis=0) * Emission[:, Observation[t]]

    last = int(np.argmax(V[:, -1]))
    P = V[last, -1]
    path = [last]
    for t in range(T - 1, 0, -1):
        last = int(back[last, t])
        path.append(last)
    path.reverse()
    return path, P
