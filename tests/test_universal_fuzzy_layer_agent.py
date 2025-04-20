import numpy as np
import pytest
from src.core.agents.agent import Agent
from src.core.neural_networks.universal_fuzzy_layer import UniversalFuzzyLayer
from src.core.neural_networks.fuzzy_system import FuzzySet

def make_agent_with_fuzzy():
    # Dummy model and policy
    model = object()
    agent = Agent(model)
    # Define fuzzy sets for 2D input
    fs0 = FuzzySet('Low', [0.0, 1.0])
    fs1 = FuzzySet('High', [1.0, 1.0])
    fuzzy_layer = UniversalFuzzyLayer(fuzzy_sets_per_input=[[fs0, fs1], [fs0, fs1]])
    agent.attach_fuzzy_layer(fuzzy_layer)
    return agent, fs0, fs1

def test_attach_and_evaluate_fuzzy_layer():
    agent, fs0, fs1 = make_agent_with_fuzzy()
    assert agent.has_fuzzy_layer()
    # Add a rule: IF x0 is Low AND x1 is High THEN output 42
    antecedents = [(0, fs0), (1, fs1)]
    agent.fuzzy_add_rule(antecedents, 42)
    # Evaluate for input [0, 1]
    x = np.array([0, 1])
    output = agent.fuzzy_evaluate(x)
    assert isinstance(output, (float, np.floating, int))

def test_fuzzy_explain_and_rule_management():
    agent, fs0, fs1 = make_agent_with_fuzzy()
    antecedents = [(0, fs0), (1, fs1)]
    agent.fuzzy_add_rule(antecedents, 7)
    explanation = agent.fuzzy_explain([0, 1])
    assert 'rule_activations' in explanation and 'output' in explanation
    rules = agent.fuzzy_list_rules()
    assert any(r['consequent'] == 7 for r in rules)
    agent.fuzzy_prune_rule(0)
    assert len(agent.fuzzy_list_rules()) == 0

def test_detach_fuzzy_layer():
    agent, _, _ = make_agent_with_fuzzy()
    agent.detach_fuzzy_layer()
    assert not agent.has_fuzzy_layer()
    with pytest.raises(AttributeError):
        agent.fuzzy_evaluate([0, 1])
