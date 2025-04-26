import numpy as np

from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin
from neuro_fuzzy_multiagent.env.base_env import BaseEnvironment


@register_plugin("environment")
class SimpleDiscreteEnv(BaseEnvironment):
    """
    Simple discrete environment for tabular Q-learning (e.g., N-state chain).
    """

    def __init__(self, n_states=5, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state = 0

    def reset(self):
        self.state = np.random.randint(self.n_states)
        return self.get_observation()

    def step(self, action):
        # Move left/right in a chain
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.n_states - 1, self.state + 1)
        reward = 1 if self.state == self.n_states - 1 else 0
        done = self.state == self.n_states - 1
        return self.get_observation(), reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")

    def get_observation(self):
        return self.state

    def perceive(self):
        return self.get_observation()

    def extract_features(self, state=None):
        import numpy as np

        if state is None:
            state = self.get_observation()
        return np.array(state)

    def get_state(self):
        return {"state": self.state}

    @property
    def action_space(self):
        return self.n_actions

    @property
    def observation_space(self):
        return self.n_states


@register_plugin("environment")
class SimpleContinuousEnv(BaseEnvironment):
    """
    Simple continuous environment for DQN (e.g., 2D point to goal).
    """

    def __init__(self):
        self.state = np.zeros(2)
        self.goal = np.ones(2)

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=2)
        return self.get_observation()

    def step(self, action):
        # Actions: 0=+x, 1=-x, 2=+y, 3=-y
        if action == 0:
            self.state[0] += 0.1
        elif action == 1:
            self.state[0] -= 0.1
        elif action == 2:
            self.state[1] += 0.1
        elif action == 3:
            self.state[1] -= 0.1
        self.state = np.clip(self.state, -1, 1)
        reward = -np.linalg.norm(self.state - self.goal)
        done = np.linalg.norm(self.state - self.goal) < 0.2
        return self.get_observation(), reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}, Goal: {self.goal}")

    def get_observation(self):
        return self.state.copy()

    def perceive(self):
        return self.get_observation()

    def extract_features(self, state=None):
        import numpy as np

        if state is None:
            state = self.get_observation()
        return np.array(state)

    def get_state(self):
        return {"state": self.state.copy(), "goal": self.goal.copy()}

    @property
    def action_space(self):
        return 4

    @property
    def observation_space(self):
        return 2
