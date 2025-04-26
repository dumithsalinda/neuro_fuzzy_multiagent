from neuro_fuzzy_multiagent.core.agents.agent import Agent


class DummyAgent(Agent):
    def __init__(self, model=None, group=None):
        super().__init__(model=model)
        self.knowledge_received = []
        self.group = group
