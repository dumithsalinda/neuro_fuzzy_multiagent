import pytest
import gym
from src.core.environments.gym_env_wrapper import GymEnvWrapper

def test_gym_env_wrapper_cartpole():
    wrapper = GymEnvWrapper('CartPole-v1')
    obs = wrapper.reset()
    assert wrapper.observation_space.shape[0] == 4
    done = False
    steps = 0
    while not done and steps < 5:
        action = wrapper.action_space.sample()
        obs, reward, done, info = wrapper.step(action)
        steps += 1
    wrapper.close()
