class BaseAgent:
    """Minimal base class for agents."""

    def __init__(self):
        pass

    def act(self, observation):
        raise NotImplementedError
