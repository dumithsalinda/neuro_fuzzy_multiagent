from src.core.agent import Agent

class DummyAgent(Agent):
    """
    A minimal agent for testing plug-and-play registration.
    Always selects action 0 and does not learn.
    """
    def __init__(self):
        super().__init__(model=None)
    def act(self, observation):
        return 0
    def learn(self, *args, **kwargs):
        pass
