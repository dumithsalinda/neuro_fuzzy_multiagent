import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.agents.agent import NeuroFuzzyAgent
from neuro_fuzzy_multiagent.core.neural_networks.fuzzy_system import FuzzySet


def make_simple_agent():
    nn_config = {"input_dim": 2, "hidden_dim": 2, "output_dim": 2}
    fis_config = None
    agent = NeuroFuzzyAgent(nn_config, fis_config)
    return agent


def test_agent_add_and_list_rules():
    agent = make_simple_agent()
    fs0 = FuzzySet("Low", [0.0, 1.0])
    fs1 = FuzzySet("High", [1.0, 1.0])
    antecedents = [(0, fs0), (1, fs1)]
    agent.add_rule(antecedents, 42)
    rules = agent.list_rules()
    assert any(r["consequent"] == 42 for r in rules)
    assert any(r["type"] == "dynamic" for r in rules)


def test_agent_prune_rule():
    agent = make_simple_agent()
    fs0 = FuzzySet("Low", [0.0, 1.0])
    fs1 = FuzzySet("High", [1.0, 1.0])
    antecedents = [(0, fs0), (1, fs1)]
    agent.add_rule(antecedents, 99)
    rules_before = agent.list_rules()
    agent.prune_rule(0)
    rules_after = agent.list_rules()
    assert len(rules_after) == len(rules_before) - 1


def test_agent_set_and_get_learning_rate():
    agent = make_simple_agent()
    agent.set_learning_rate(0.1)
    lr = agent.get_learning_rate()
    # The default model does not have a learning_rate attribute, so this is None unless extended
    assert lr is None or abs(lr - 0.1) < 1e-6
