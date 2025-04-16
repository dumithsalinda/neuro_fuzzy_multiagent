import numpy as np
import pytest
from src.core.agent import Agent, NeuroFuzzyAgent
from src.core.neuro_fuzzy import NeuroFuzzyHybrid

def test_agent_learn_from_url(monkeypatch):
    # Simulate a web resource
    dummy_text = "policy: always return 42"
    class DummyResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass
    def dummy_get(url):
        return DummyResponse(dummy_text)
    monkeypatch.setattr("requests.get", dummy_get)
    # Agent with a model that can be updated
    class Model:
        def __init__(self):
            self.policy_str = None
        def update_from_knowledge(self, knowledge):
            self.policy_str = knowledge
    model = Model()
    agent = Agent(model)
    agent.learn_from_url("http://dummy.url/policy.txt")
    assert hasattr(agent.model, "policy_str")
    assert agent.model.policy_str == dummy_text

def test_neurofuzzyagent_learn_from_url(monkeypatch):
    dummy_text = "fuzzy_rule: IF x > 0 THEN y = 1"
    class DummyResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass
    def dummy_get(url):
        return DummyResponse(dummy_text)
    monkeypatch.setattr("requests.get", dummy_get)
    # Patch NeuroFuzzyHybrid to add update_from_knowledge
    class PatchedNFH(NeuroFuzzyHybrid):
        def __init__(self, nn_cfg, fis_cfg):
            super().__init__(nn_cfg, fis_cfg)
            self.fetched = None
        def update_from_knowledge(self, knowledge):
            self.fetched = knowledge
    # Provide minimal valid config
    nn_config = {'input_dim': 1, 'hidden_dim': 1, 'output_dim': 1}
    fis_config = {}
    agent = NeuroFuzzyAgent(nn_config=nn_config, fis_config=fis_config)
    agent.model = PatchedNFH(nn_config, fis_config)
    agent.learn_from_url("http://dummy.url/fuzzy.txt")
    assert hasattr(agent.model, "fetched")
    assert agent.model.fetched == dummy_text
