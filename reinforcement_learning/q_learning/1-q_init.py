#!/usr/bin/env python3
"""Load a FrozenLake environment using OpenAI Gym."""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the pre-made FrozenLakeEnv environment from OpenAI's gym.

    Args:
        desc: None or a list of lists describing a custom map.
        map_name: None or a string naming a pre-made map.
        is_slippery: boolean indicating whether the ice is slippery.

    Returns:
        env: the initialized Gym environment instance.
    """
    env = gym.make(
        'FrozenLake-v0',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
