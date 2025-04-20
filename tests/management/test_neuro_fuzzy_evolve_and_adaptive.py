import numpy as np
import pytest
from src.core.agents.agent import NeuroFuzzyAgent
from src.core.neural_networks.fuzzy_system import FuzzySet

def make_agent_with_rules():
    nn_config = {'input_dim': 2, 'hidden_dim': 2, 'output_dim': 1}
    agent = NeuroFuzzyAgent(nn_config, None)
    # Add two fuzzy sets per input
    fs0 = FuzzySet('Low', [0.0, 1.0])
    fs1 = FuzzySet('High', [1.0, 1.0])
    # Add two dynamic rules
    agent.add_rule([(0, fs0), (1, fs1)], 10)
    agent.add_rule([(0, fs1), (1, fs0)], 5)
    return agent, [fs0, fs1]

def test_evolve_rules_pruning():
    agent, fuzzy_sets = make_agent_with_rules()
    # Inputs that activate only the first rule
    recent_inputs = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
    # Prune rules with avg activation < 0.5
    pruned = agent.evolve_rules(recent_inputs=recent_inputs, min_avg_activation=0.5)
    rules = agent.list_rules()
    # Only one rule should remain
    assert pruned
    assert len([r for r in rules if r['type'] == 'dynamic']) == 1

def test_auto_switch_mode():
    agent, _ = make_agent_with_rules()
    # Simulate error history
    # Start in 'hybrid', error is high, should switch to 'fuzzy'
    agent.set_mode('hybrid')
    mode1 = agent.auto_switch_mode(error_history=[0.3, 0.25, 0.2, 0.18, 0.16])
    assert mode1 == 'fuzzy'
    # Now error is high in fuzzy, should switch to 'neural'
    agent.set_mode('fuzzy')
    mode2 = agent.auto_switch_mode(error_history=[0.3, 0.25, 0.2, 0.18, 0.16])
    assert mode2 == 'neural'
    # Low error, should stay in neural
    agent.set_mode('neural')
    mode3 = agent.auto_switch_mode(error_history=[0.05, 0.04, 0.03, 0.02, 0.01])
    assert mode3 == 'neural'
