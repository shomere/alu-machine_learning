#!/usr/bin/env python3
"""Policy gradient utilities."""

import numpy as np


def policy(matrix, weight):
    """Compute a stochastic policy using softmax over linear logits.

    Args:
        matrix: numpy.ndarray representing the state/observation.
        weight: numpy.ndarray of weights.

    Returns:
        numpy.ndarray of action probabilities.
    """
    z = np.dot(matrix, weight)
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """Compute an action and the gradient of log-policy wrt weight.

    Args:
        state: numpy.ndarray representing the current observation.
        weight: numpy.ndarray of weights.

    Returns:
        tuple: (action, grad) where action is an int and grad is a
            numpy.ndarray matching weight's shape.
    """
    probs = policy(state, weight)
    action_probs = probs[0]
    action = int(np.random.choice(len(action_probs), p=action_probs))

    one_hot = np.zeros_like(probs)
    one_hot[..., action] = 1.0

    grad = np.dot(state.T, one_hot - probs)
    return action, grad
