#!/usr/bin/env python3
"""Run a trained agent on a FrozenLake episode."""

import numpy as np


def play(env, Q, max_steps=100):
    """Have a trained agent play one episode by exploiting Q.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the trained Q-table.
        max_steps: maximum number of steps in the episode.

    Returns:
        float: total reward accumulated during the episode.
    """
    state = env.reset()
    total_reward = 0.0
    env.render()

    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        state = next_state

        if done:
            break

    return total_reward
