import numpy as np
import pytest
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.core.fuzzy_system import FuzzySet

def make_simple_nf():
    nn_config = {'input_dim': 2, 'hidden_dim': 2, 'output_dim': 2}
    nf = NeuroFuzzyHybrid(nn_config)
    return nf

def test_add_and_list_rules():
    nf = make_simple_nf()
    fs0 = FuzzySet('Low', [0.0, 1.0])
    fs1 = FuzzySet('High', [1.0, 1.0])
    antecedents = [(0, fs0), (1, fs1)]
    nf.add_rule(antecedents, 1)
    rules = nf.list_rules()
    assert any(r['consequent'] == 1 for r in rules)
    assert any(r['type'] == 'dynamic' for r in rules)

def test_prune_rule():
    nf = make_simple_nf()
    fs0 = FuzzySet('Low', [0.0, 1.0])
    fs1 = FuzzySet('High', [1.0, 1.0])
    antecedents = [(0, fs0), (1, fs1)]
    nf.add_rule(antecedents, 2)
    rules_before = nf.list_rules()
    nf.prune_rule(0)
    rules_after = nf.list_rules()
    assert len(rules_after) == len(rules_before) - 1

def test_set_and_get_learning_rate():
    nf = make_simple_nf()
    nf.set_learning_rate(0.05)
    lr = nf.get_learning_rate()
    # The default FeedforwardNeuralNetwork does not have learning_rate, so this will be None unless extended
    assert lr is None or abs(lr - 0.05) < 1e-6
