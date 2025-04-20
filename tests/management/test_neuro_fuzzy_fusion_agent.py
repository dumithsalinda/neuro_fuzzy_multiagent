import numpy as np
import pytest
from src.core.agents.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent

class DummyFuzzySystem:
    """Minimal dummy fuzzy system for testing."""
    def __init__(self, output_dim=3):
        self.output_dim = output_dim
    def infer(self, obs_list):
        # Return a constant or deterministic vector for testing
        return np.ones(self.output_dim) * 2.0
    def evaluate(self, obs_vector):
        # Return a constant vector for compatibility with agent.act
        return np.ones(self.output_dim) * 2.0
    def explain_inference(self, obs_list):
        return {'output': self.infer(obs_list), 'rules': ['dummy_rule']}

@pytest.fixture
def agent():
    # 2 modalities, each dim=4, output_dim=3
    input_dims = [4, 4]
    hidden_dim = 8
    output_dim = 3
    agent = NeuroFuzzyFusionAgent(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        fusion_type='concat',
        fuzzy_config=None,
        fusion_alpha=0.5
    )
    # Patch fuzzy_system with dummy
    agent.fuzzy_system = DummyFuzzySystem(output_dim)
    return agent

def test_act_and_explain(agent):
    obs1 = np.random.randn(4)
    obs2 = np.random.randn(4)
    obs_list = [obs1, obs2]
    action = agent.act(obs_list)
    assert isinstance(action, int)
    # Check explainability
    exp = agent.explain_action(obs_list)
    assert 'neural_output' in exp
    assert 'fuzzy_explanation' in exp
    assert 'fused_output' in exp
    assert 'chosen_action' in exp
    assert len(exp['fused_output']) == 3

def test_train_step(agent):
    obs1 = np.random.randn(4)
    obs2 = np.random.randn(4)
    obs_list = [obs1, obs2]
    target = np.array([1.0, 2.0, 3.0])
    loss = agent.train_step(obs_list, target)
    assert isinstance(loss, float)
    assert loss >= 0.0
