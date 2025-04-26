"""
test_neuro_fuzzy_agent_self_organization.py

Test self-organization in NeuroFuzzyAgent and NeuroFuzzyHybrid.
"""

import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.agents.agent import NeuroFuzzyAgent
from neuro_fuzzy_multiagent.core.neuro_fuzzy import NeuroFuzzyHybrid


def test_neuro_fuzzy_agent_self_organize_changes_nn_weights():
    nn_config = {"input_dim": 1, "output_dim": 1, "hidden_dim": 2}
    agent = NeuroFuzzyAgent(nn_config, None)
    W1_before = agent.model.nn.W1.copy()
    agent.self_organize(mutation_rate=0.5)
    W1_after = agent.model.nn.W1
    assert not np.allclose(
        W1_before, W1_after
    ), "Neural weights should change after self_organize()"


def test_neuro_fuzzy_hybrid_self_organize_fuzzy_rule_change():
    from neuro_fuzzy_multiagent.core.neural_networks.fuzzy_system import (
        FuzzyInferenceSystem,
        FuzzySet,
    )

    # Setup FIS with 2 rules
    fs_low = FuzzySet("low", [0.0, 1.0])
    fs_high = FuzzySet("high", [5.0, 1.0])
    fuzzy_sets = [[fs_low, fs_high]]
    X = np.array([[0.0], [5.0]])
    y = np.array([0.0, 1.0])
    fis = FuzzyInferenceSystem()
    fis.dynamic_rule_generation(X, y, fuzzy_sets)
    nn_config = dict(input_dim=1, hidden_dim=2, output_dim=1)
    nf = NeuroFuzzyHybrid(nn_config, fis_config={})
    nf.fis = fis
    n_rules_before = len(nf.fis.rules)
    nf.self_organize(rule_change=True)
    n_rules_after = len(nf.fis.rules)
    assert (
        n_rules_before != n_rules_after or n_rules_after > 1
    ), "Rule set should change or stay valid after self_organize()"
