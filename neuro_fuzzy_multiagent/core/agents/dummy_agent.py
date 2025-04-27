from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("agent")
class DummyAgent(Agent):
    """
    A minimal agent for testing plug-and-play registration.
    Always selects action 0 and does not learn.
    """

    def __init__(self, **kwargs):
        super().__init__(model=None)
        self.extra_args = kwargs

    def act(self, observation):
        return 0

    def learn(self, *args, **kwargs):
        pass
