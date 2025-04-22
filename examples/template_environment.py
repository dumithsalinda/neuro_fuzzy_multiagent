"""
Template Environment Plugin
--------------------------
Subclass BaseEnvironment and use the @register_plugin('environment') decorator (already on base class).
"""
from src.env.base_env import BaseEnvironment

class MyTemplateEnv(BaseEnvironment):
    """
    Example environment for plug-and-play system.
    Configurable via dashboard/config.
    """
    def __init__(self, param1=0):
        self.param1 = param1
        self.state = 0
    def reset(self):
        self.state = 0
        return self.get_observation()
    def step(self, action):
        self.state += action
        return self.get_observation(), 0.0, False, {}
    def get_observation(self):
        return self.state
    def get_state(self):
        return {'state': self.state}
    def render(self, mode="human"):
        print(f"State: {self.state}")
