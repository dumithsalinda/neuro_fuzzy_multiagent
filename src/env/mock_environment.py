from src.env.base_env import BaseEnvironment


class MockEnvironment(BaseEnvironment):
    """A mock environment for fast, deterministic testing."""

    def __init__(self):
        super().__init__()
        self.state = 0

    @property
    def action_space(self):
        return [0, 1]

    @property
    def observation_space(self):
        return [0, 1, 2, 3, 4]

    def get_state(self):
        return self.state

    def get_observation(self):
        return self.state

    def render(self):
        return f"State: {self.state}"

    def step(self, action):
        self.state += 1
        return self.state, 1.0, self.state > 3, {}

    def reset(self):
        self.state = 0
        return self.state
