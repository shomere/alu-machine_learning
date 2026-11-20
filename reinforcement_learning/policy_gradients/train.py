#!/usr/bin/env python3
"""Training loop for Monte-Carlo policy gradient (REINFORCE)."""

import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train a policy using Monte-Carlo policy gradient.

    Args:
        env: OpenAI Gym environment instance.
        nb_episodes: number of episodes used for training.
        alpha: learning rate.
        gamma: discount factor.
        show_result: if True, render the environment every 1000 episodes.

    Returns:
        list: total reward (score) earned in each episode.
    """
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    weight = np.random.rand(n_features, n_actions)
    scores = []

    for episode in range(nb_episodes):
        obs = env.reset()
        state = obs[None, :]
        render_ep = show_result and (episode % 1000 == 0)

        trajectory = []
        done = False
        while not done:
            if render_ep:
                env.render()
            action, grad = policy_gradient(state, weight)
            next_obs, reward, done, _ = env.step(action)
            trajectory.append((grad, reward))
            state = next_obs[None, :]

        if render_ep:
            env.render()

        episode_score = sum(item[1] for item in trajectory)
        scores.append(episode_score)

        # Compute discounted returns G_t from the end of the episode
        returns = np.zeros(len(trajectory), dtype=np.float64)
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            G = trajectory[t][1] + gamma * G
            returns[t] = G

        # Update weights along the trajectory
        for t, (grad, _) in enumerate(trajectory):
            weight += alpha * returns[t] * grad

        print("Episode: {} Score: {}".format(episode, episode_score),
              end="\r", flush=False)

    return scores
