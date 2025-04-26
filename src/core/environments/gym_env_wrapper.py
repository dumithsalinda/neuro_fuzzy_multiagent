import gym

from src.core.plugins.registration_utils import register_plugin


@register_plugin("environment")
class GymEnvWrapper:
    """
    Wraps an OpenAI Gym environment to make it compatible with the Neuro-Fuzzy Multi-Agent platform.
    """

    def __init__(self, env_name, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
