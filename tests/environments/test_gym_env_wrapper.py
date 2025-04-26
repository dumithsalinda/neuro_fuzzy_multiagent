import pytest
import gym
from src.core.environments.gym_env_wrapper import GymEnvWrapper


def test_gym_env_wrapper_cartpole():
    wrapper = GymEnvWrapper("CartPole-v1")
    obs = wrapper.reset()
    assert wrapper.observation_space.shape[0] == 4
    done = False
    steps = 0
    while not done and steps < 5:
        action = wrapper.action_space.sample()
        result = wrapper.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        steps += 1
    wrapper.close()
