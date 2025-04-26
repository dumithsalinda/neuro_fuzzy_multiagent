import numpy as np
from src.core.agents.agent import NeuroFuzzyAgent
from src.core.management.meta_controller import MetaController
from src.core.neural_networks.fuzzy_system import FuzzySet


def make_agent():
    nn_config = {"input_dim": 2, "hidden_dim": 2, "output_dim": 1}
    agent = NeuroFuzzyAgent(nn_config, None)
    fs0 = FuzzySet("Low", [0.0, 1.0])
    fs1 = FuzzySet("High", [1.0, 1.0])
    agent.add_rule([(0, fs0), (1, fs1)], 10)
    agent.add_rule([(0, fs1), (1, fs0)], 5)
    return agent, [fs0, fs1]


def test_tune_fuzzy_rules():
    agent, fuzzy_sets = make_agent()
    meta = MetaController()
    # All data near 1.0, should shift centers
    data = [np.array([1.0, 1.0]), np.array([1.1, 1.2]), np.array([0.9, 0.95])]
    # Before tuning
    centers_before = [fs.params[0] for fs in fuzzy_sets]
    meta.tune_fuzzy_rules(agent, data)
    centers_after = [fs.params[0] for fs in fuzzy_sets]
    assert any(abs(a - b) > 0.01 for a, b in zip(centers_before, centers_after))


def test_tune_learning_rate():
    agent, _ = make_agent()
    meta = MetaController()
    meta.tune_learning_rate(agent, 0.005)
    assert agent.get_learning_rate() == 0.005


def test_embedded_meta_controller():
    agent, _ = make_agent()
    # Use agent's embedded meta_controller
    data = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    agent.meta_adapt(data=data, new_lr=0.002)
    assert agent.get_learning_rate() == 0.002
