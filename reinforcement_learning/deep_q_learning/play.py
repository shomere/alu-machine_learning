#!/usr/bin/env python3
"""Load a trained DQN policy from policy.h5 and play Atari Breakout."""

import time
import gym
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariPreprocessor:
    """Resize and convert RGB Atari frames to grayscale."""

    def __init__(self, shape=INPUT_SHAPE):
        """Initialize preprocessor with target (height, width)."""
        self.shape = shape

    def process_observation(self, obs):
        """Convert raw frame to uint8 (84, 84) grayscale image."""
        if isinstance(obs, tuple):
            obs = obs[0]
        img = Image.fromarray(obs)
        img = img.resize(self.shape, Image.BILINEAR).convert('L')
        return np.array(img, dtype='uint8')


def build_model(num_actions, window_length=WINDOW_LENGTH,
                input_shape=INPUT_SHAPE):
    """Build the DeepMind DQN convolutional neural network."""
    model = Sequential()
    model.add(Permute((2, 3, 1),
                      input_shape=(window_length,) + input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


def reset_env(env):
    """Handle Gym v0.21 and newer tuple reset signatures."""
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res


def step_env(env, action):
    """Normalize Gym step outputs to (obs, reward, done, info)."""
    res = env.step(action)
    if len(res) == 5:
        obs, reward, terminated, truncated, info = res
        return obs, reward, bool(terminated or truncated), info
    return res


def main():
    """Load policy.h5 and render episodes of Breakout with GreedyQPolicy."""
    env = gym.make('BreakoutDeterministic-v4')
    num_actions = env.action_space.n
    preprocessor = AtariPreprocessor(INPUT_SHAPE)

    model = build_model(num_actions)
    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=10000
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')

    class RenderedBreakoutEnv(gym.Wrapper):
        """Gym wrapper that renders frames and preprocesses observations."""

        def reset(self, **kwargs):
            """Reset environment, render frame, and preprocess."""
            obs = reset_env(self.env)
            self.env.render()
            return preprocessor.process_observation(obs)

        def step(self, action):
            """Step environment, render frame, and preprocess."""
            obs, reward, done, info = step_env(self.env, action)
            self.env.render()
            time.sleep(0.01)
            return (
                preprocessor.process_observation(obs),
                reward,
                done,
                info
            )

    rendered_env = RenderedBreakoutEnv(env)
    dqn.test(rendered_env, nb_episodes=5, visualize=False)
    env.close()


if __name__ == '__main__':
    main()
