#!/usr/bin/env python3
"""SARSA(lambda) control algorithm with eligibility traces."""

import numpy as np


def _epsilon_greedy(Q, state, epsilon, n_actions):
    """Select an action using an epsilon-greedy policy."""
    if np.random.uniform() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm to update action values.

    Args:
        env: OpenAI Gym environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q-table.
        lambtha: eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.
        epsilon: initial threshold for epsilon-greedy exploration.
        min_epsilon: minimum value that epsilon should decay to.
        epsilon_decay: decay rate for updating epsilon between episodes.

    Returns:
        Q: the updated Q-table numpy.ndarray.
    """
    n_actions = Q.shape[1]
    init_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()
        action = _epsilon_greedy(Q, state, epsilon, n_actions)
        eligibilities = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = _epsilon_greedy(Q, next_state, epsilon, n_actions)

            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            eligibilities = gamma * lambtha * eligibilities
            eligibilities[state, action] += 1

            Q += alpha * delta * eligibilities
            state = next_state
            action = next_action

            if done:
                break

        epsilon = max(min_epsilon, init_epsilon * np.exp(-epsilon_decay * ep))

    return Q
