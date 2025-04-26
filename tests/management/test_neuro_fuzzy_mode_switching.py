import numpy as np
import pytest
from src.core.agents.agent import NeuroFuzzyAgent
from src.core.neural_networks.fuzzy_system import FuzzySet


def make_nf_agent():
    nn_config = {"input_dim": 2, "hidden_dim": 2, "output_dim": 1}
    fis_config = None
    agent = NeuroFuzzyAgent(nn_config, fis_config)
    # Add fuzzy sets and a rule
    fs0 = FuzzySet("Low", [0.0, 1.0])
    fs1 = FuzzySet("High", [1.0, 1.0])
    antecedents = [(0, fs0), (1, fs1)]
    agent.add_rule(antecedents, 10)
    return agent, fs0, fs1


def test_mode_switching_and_inference():
    agent, fs0, fs1 = make_nf_agent()
    x = np.array([0, 1])
    # Fuzzy only
    agent.set_mode("fuzzy")
    fuzzy_out = agent.model.fis.evaluate(x)
    out_fuzzy = agent.model.forward(x)
    assert np.allclose(out_fuzzy, fuzzy_out)
    # Neural only
    agent.set_mode("neural")
    nn_out = agent.model.nn.forward(x)
    out_nn = agent.model.forward(x)
    assert np.allclose(out_nn, nn_out)
    # Hybrid (default 0.5 weight)
    agent.set_mode("hybrid")
    hybrid_out = agent.model.forward(x)
    expected = 0.5 * nn_out + 0.5 * fuzzy_out
    assert np.allclose(hybrid_out, expected)
    # Hybrid (custom weight)
    agent.set_mode("hybrid", hybrid_weight=0.8)
    hybrid_out2 = agent.model.forward(x)
    expected2 = 0.8 * nn_out + 0.2 * fuzzy_out
    assert np.allclose(hybrid_out2, expected2)


def test_runtime_switching():
    agent, fs0, fs1 = make_nf_agent()
    x = np.array([0, 1])
    # Switch modes at runtime
    agent.set_mode("fuzzy")
    out1 = agent.model.forward(x)
    agent.set_mode("neural")
    out2 = agent.model.forward(x)
    agent.set_mode("hybrid")
    out3 = agent.model.forward(x)
    # All should be different in general
    assert not np.allclose(out1, out2)
    assert not np.allclose(out2, out3)
    assert not np.allclose(out1, out3)
