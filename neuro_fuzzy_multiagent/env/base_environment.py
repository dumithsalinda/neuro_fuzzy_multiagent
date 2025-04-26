class BaseEnvironment:
    """Minimal base class for environments."""

    def __init__(self):
        pass

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
