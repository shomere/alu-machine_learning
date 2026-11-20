#!/usr/bin/env python3
"""Epsilon-greedy action selection policy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Determine the next action using an epsilon-greedy strategy.

    Args:
        Q: numpy.ndarray containing the Q-table.
        state: current state index.
        epsilon: exploration probability threshold.

    Returns:
        int: index of the next action to take.
    """
    p = np.random.uniform()
    if p < epsilon:
        return int(np.random.randint(Q.shape[1]))
    return int(np.argmax(Q[state]))
