#!/usr/bin/env python3
"""The Baum-Welch algorithm for a hidden markov model"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model

    Observations: numpy.ndarray of shape (T,), observation indices
    Transition: numpy.ndarray of shape (M, M), transition probabilities
    Emission: numpy.ndarray of shape (M, N), emission probabilities
    Initial: numpy.ndarray of shape (M, 1), starting probabilities
    iterations: number of times expectation-maximization is performed
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    T = Observations.shape[0]
    M, N = Emission.shape
    if Transition.shape != (M, M) or Initial.shape != (M, 1):
        return None, None
    if np.any(Observations < 0) or np.any(Observations >= N):
        return None, None

    A = Transition.copy()
    E = Emission.copy()
    # one-hot encoding of the observations, shape (T, N)
    onehot = (Observations[:, np.newaxis] == np.arange(N)).astype(float)

    for _ in range(iterations):
        alpha = np.zeros((M, T))
        beta = np.zeros((M, T))
        alpha[:, 0] = Initial[:, 0] * E[:, Observations[0]]
        for t in range(1, T):
            alpha[:, t] = np.matmul(alpha[:, t - 1], A) * \
                E[:, Observations[t]]
        beta[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            beta[:, t] = np.matmul(
                A, E[:, Observations[t + 1]] * beta[:, t + 1])

        # xi[i, j, t]: probability of state i at t and state j at t + 1
        xi = alpha[:, np.newaxis, :T - 1] * A[:, :, np.newaxis] * \
            (E[:, Observations[1:]] * beta[:, 1:])[np.newaxis, :, :]
        xi = xi / np.sum(xi, axis=(0, 1))
        # gamma[i, t]: probability of state i at time t
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=0)

        A = np.sum(xi, axis=2) / np.sum(gamma[:, :T - 1], axis=1,
                                        keepdims=True)
        E = np.matmul(gamma, onehot) / np.sum(gamma, axis=1, keepdims=True)

    return A, E
