import unittest
import numpy as np
from src.core.agents.anfis_hybrid import ANFISHybrid

class TestANFISHybrid(unittest.TestCase):
    def setUp(self):
        self.model = ANFISHybrid(input_dim=2, n_rules=3)

    def test_forward_shape(self):
        x = np.array([0.5, -0.2])
        y = self.model.forward(x)
        self.assertIsInstance(y, float)

    def test_update_learns(self):
        x = np.array([0.0, 0.0])
        y_target = 1.0
        y_before = self.model.forward(x)
        for _ in range(200):
            self.model.update(x, y_target, lr=0.5)
        y_after = self.model.forward(x)
        self.assertTrue(abs(y_after - y_target) < abs(y_before - y_target))

    def test_add_and_remove_rule(self):
        n_before = self.model.n_rules
        self.model.add_rule(center=np.zeros((1,2)), width=np.ones((1,2)), weight=0.0)
        self.assertEqual(self.model.n_rules, n_before + 1)
        self.model.remove_rule(self.model.n_rules - 1)
        self.assertEqual(self.model.n_rules, n_before)

    def test_dynamic_rule_update(self):
        # Artificially set firing and error histories
        self.model.firing_history[0] = [0.0]*25
        self.model.error_history[0] = [2.0]*25
        self.model._last_firing_strengths = np.zeros(self.model.n_rules)
        self.model._last_input = np.array([1.0, 1.0])
        n_before = self.model.n_rules
        self.model.dynamic_rule_update(error_threshold=1.0, firing_threshold=0.1, min_history=20)
        self.assertTrue(self.model.n_rules > n_before or self.model.n_rules < n_before)

if __name__ == "__main__":
    unittest.main()
