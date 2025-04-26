from neuro_fuzzy_multiagent.agents.base_agent import BaseAgent


class MockAgent(BaseAgent):
    """A mock agent for fast, deterministic testing."""

    def __init__(self):
        super().__init__()

    def act(self, observation):
        return 0
