import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from neuro_fuzzy_multiagent.core.neural_networks.fuzzy_system import (
    FuzzyInferenceSystem,
    FuzzyRule,
    FuzzySet,
)


def test_fuzzy_set_membership_and_tune():
    fs = FuzzySet("A", [0.0, 1.0])
    # Membership at center should be ~1
    assert np.isclose(fs.membership(0.0), 1.0)
    # Membership far from center should be small
    assert fs.membership(10.0) < 1e-4
    # Self-tune: update center/width
    data = np.array([2.0, 2.1, 1.9, 2.05])
    fs.tune(data)
    assert np.isclose(fs.params[0], np.mean(data))
    assert np.isclose(fs.params[1], np.std(data) + 1e-6)


def test_dynamic_rule_generation_and_evaluate():
    # Two fuzzy sets per input
    fs_low = FuzzySet("low", [0.0, 1.0])
    fs_high = FuzzySet("high", [5.0, 1.0])
    fuzzy_sets = [[fs_low, fs_high]]  # 1D input
    # Toy data: X = [[0], [5]], y = [0, 1]
    X = np.array([[0.0], [5.0]])
    y = np.array([0.0, 1.0])
    fis = FuzzyInferenceSystem()
    fis.dynamic_rule_generation(X, y, fuzzy_sets)
    # Evaluate near 0 should be near 0, near 5 should be near 1
    assert fis.evaluate([0.0]) < 0.2
    assert fis.evaluate([5.0]) > 0.8
    # In between should interpolate
    mid_val = fis.evaluate([2.5])
    assert 0.2 < mid_val < 0.8
