from src.core.agents.agent import Agent

from src.core.plugins.registration_utils import register_plugin


@register_plugin("agent")
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
