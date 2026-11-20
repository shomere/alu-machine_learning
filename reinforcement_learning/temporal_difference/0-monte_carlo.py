#!/usr/bin/env python3
"""Monte Carlo prediction algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the first-visit Monte Carlo algorithm to estimate V.

    Args:
        env: OpenAI Gym environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: function taking a state and returning the next action.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.

    Returns:
        V: the updated value estimate numpy.ndarray.
    """
    for _ in range(episodes):
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        visited = [x[0] for x in episode]
        for t in range(len(episode) - 1, -1, -1):
            state_t, reward_t = episode[t]
            G = gamma * G + reward_t
            if state_t not in visited[:t]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
