#!/usr/bin/env python3
"""TD(lambda) prediction algorithm with eligibility traces."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform the TD(lambda) algorithm to estimate state values.

    Args:
        env: OpenAI Gym environment instance.
        V: numpy.ndarray of shape (s,) containing value estimates.
        policy: function taking a state and returning the next action.
        lambtha: eligibility trace factor.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.

    Returns:
        V: the updated value estimate numpy.ndarray.
    """
    for _ in range(episodes):
        state = env.reset()
        eligibilities = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibilities = gamma * lambtha * eligibilities
            eligibilities[state] += 1

            V += alpha * delta * eligibilities
            state = next_state

            if done:
                break

    return V
