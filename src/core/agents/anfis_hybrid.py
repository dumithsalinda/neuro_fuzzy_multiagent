import numpy as np


class ANFISHybrid:
    """
    Minimal Adaptive Neuro-Fuzzy Inference System (ANFIS) hybrid model.
    - Supports Gaussian membership functions (tunable params)
    - Rule weights are learnable (simple gradient update placeholder)
    - Forward: computes fuzzy rule firing strengths, weighted sum for output
    - Dynamic: can add/prune rules based on usage and error
    """

    def __init__(self, input_dim, n_rules):
        self.input_dim = input_dim
        self.n_rules = n_rules
        # Each rule has a center and width per input dimension
        self.centers = np.random.randn(n_rules, input_dim)
        self.widths = np.abs(np.random.randn(n_rules, input_dim)) + 1e-2
        self.rule_weights = np.random.randn(n_rules)
        # For dynamic rule management
        self.firing_history = [[] for _ in range(n_rules)]  # recent firing strengths
        self.error_history = [[] for _ in range(n_rules)]  # recent errors
        self.max_history = 50

    def membership(self, x, center, width):
        # Gaussian membership
        return np.exp(-((x - center) ** 2) / (2 * width**2))

    def forward(self, x):
        x = np.asarray(x)
        firing_strengths = np.ones(self.n_rules)
        for r in range(self.n_rules):
            for d in range(self.input_dim):
                firing_strengths[r] *= self.membership(
                    x[d], self.centers[r, d], self.widths[r, d]
                )
        # Store firing strengths for rule usage tracking
        self._last_firing_strengths = firing_strengths.copy()
        if firing_strengths.sum() == 0:
            output = 0.0
        else:
            output = (
                np.dot(firing_strengths, self.rule_weights) / firing_strengths.sum()
            )
        return output

    def update(self, x, target, lr=0.01):
        # Self-tuning: gradient update for rule weights, centers, and widths
        pred = self.forward(x)
        error = target - pred
        x = np.asarray(x)
        firing_strengths = np.ones(self.n_rules)
        memberships = np.zeros((self.n_rules, self.input_dim))
        for r in range(self.n_rules):
            for d in range(self.input_dim):
                memberships[r, d] = self.membership(
                    x[d], self.centers[r, d], self.widths[r, d]
                )
                firing_strengths[r] *= memberships[r, d]
        norm = firing_strengths.sum() if firing_strengths.sum() != 0 else 1.0
        grad = firing_strengths / norm
        self.rule_weights += lr * error * grad
        # Gradient for centers and widths (chain rule)
        for r in range(self.n_rules):
            for d in range(self.input_dim):
                mu = memberships[r, d]
                c = self.centers[r, d]
                w = self.widths[r, d]
                # Partial derivative of output w.r.t. center
                if firing_strengths[r] != 0:
                    dmu_dc = mu * (x[d] - c) / (w**2)
                    d_out_dc = (
                        self.rule_weights[r]
                        * (dmu_dc * firing_strengths[r] / mu)
                        * norm
                        - self.rule_weights
                        @ firing_strengths
                        * dmu_dc
                        * firing_strengths[r]
                        / (mu * norm)
                    ) / (norm**2)
                    self.centers[r, d] += lr * error * d_out_dc
                    # Partial derivative of output w.r.t. width
                    dmu_dw = mu * ((x[d] - c) ** 2) / (w**3)
                    d_out_dw = (
                        self.rule_weights[r]
                        * (dmu_dw * firing_strengths[r] / mu)
                        * norm
                        - self.rule_weights
                        @ firing_strengths
                        * dmu_dw
                        * firing_strengths[r]
                        / (mu * norm)
                    ) / (norm**2)
                    self.widths[r, d] += lr * error * d_out_dw
                    # Clamp widths to avoid collapse
                    self.widths[r, d] = max(self.widths[r, d], 1e-3)
        # --- Track firing and error history for dynamic rule management ---
        for r in range(self.n_rules):
            if len(self.firing_history[r]) >= self.max_history:
                self.firing_history[r].pop(0)
            self.firing_history[r].append(firing_strengths[r])
            if len(self.error_history[r]) >= self.max_history:
                self.error_history[r].pop(0)
            self.error_history[r].append(abs(error))

    def add_rule(self, center, width, weight=0.0):
        self.centers = np.vstack([self.centers, center])
        self.widths = np.vstack([self.widths, width])
        self.rule_weights = np.append(self.rule_weights, weight)
        self.n_rules += 1
        # Expand histories
        self.firing_history.append([])
        self.error_history.append([])

    def remove_rule(self, idx):
        if self.n_rules <= 1:
            return
        self.centers = np.delete(self.centers, idx, axis=0)
        self.widths = np.delete(self.widths, idx, axis=0)
        self.rule_weights = np.delete(self.rule_weights, idx)
        del self.firing_history[idx]
        del self.error_history[idx]
        self.n_rules -= 1

    def prune_rules(self, threshold=1e-3, min_history=20):
        # Remove rules with consistently low firing strength
        to_remove = []
        for i, history in enumerate(self.firing_history):
            if len(history) >= min_history and np.mean(history) < threshold:
                to_remove.append(i)
        for idx in reversed(to_remove):
            self.remove_rule(idx)

    def generate_rule(self, x, width=0.5, weight=0.0):
        # Add a new rule centered at x if high error persists
        center = np.array(x).reshape(1, -1)
        width_arr = np.ones((1, self.input_dim)) * width
        self.add_rule(center, width_arr, weight)

    def dynamic_rule_update(
        self, error_threshold=0.5, firing_threshold=1e-3, min_history=20
    ):
        # Prune underused rules
        self.prune_rules(threshold=firing_threshold, min_history=min_history)
        # Generate rule if recent error is high and no rule fires strongly
        recent_errors = [
            np.mean(hist[-min_history:]) if len(hist) >= min_history else 0
            for hist in self.error_history
        ]
        if len(recent_errors) > 0 and max(recent_errors) > error_threshold:
            if (
                hasattr(self, "_last_firing_strengths")
                and max(self._last_firing_strengths) < firing_threshold
            ):
                if hasattr(self, "_last_input"):
                    self.generate_rule(self._last_input)
