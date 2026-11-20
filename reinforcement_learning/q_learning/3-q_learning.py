#!/usr/bin/env python3
"""Q-learning training algorithm."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning to train a Q-table.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.
        epsilon: initial threshold for epsilon-greedy exploration.
        min_epsilon: minimum value that epsilon should decay to.
        epsilon_decay: decay rate for updating epsilon between episodes.

    Returns:
        tuple: (Q, total_rewards) where Q is the updated Q-table and
            total_rewards is a list of total rewards per episode.
    """
    initial_epsilon = epsilon
    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            if done and reward == 0:
                reward = -1.0

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)
        epsilon = max(min_epsilon, initial_epsilon * np.exp(-epsilon_decay * ep))

    return Q, total_rewards
