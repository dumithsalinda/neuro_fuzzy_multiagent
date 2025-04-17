import numpy as np
try:
    from .anfis_hybrid import ANFISHybrid
except ImportError:
    from anfis_hybrid import ANFISHybrid

class NeuroFuzzyANFISAgent:
    """
    Agent wrapper for the ANFISHybrid neuro-fuzzy model.
    Supports act (forward) and observe (online update) methods.
    """
    def __init__(self, input_dim, n_rules, lr=0.01):
        self.model = ANFISHybrid(input_dim, n_rules)
        self.lr = lr
        self.last_obs = None
        self.last_action = None

    def act(self, obs):
        # Forward pass through ANFIS
        self.last_obs = obs
        action = self.model.forward(obs)
        self.last_action = action
        return action

    def observe(self, obs, reward):
        # Online update: treat reward as target
        self.model.update(self.last_obs, reward, lr=self.lr)

# Simple test/demo
if __name__ == "__main__":
    # Regression: y = x0 + x1 (target)
    agent = NeuroFuzzyANFISAgent(input_dim=2, n_rules=4, lr=0.05)
    for step in range(200):
        x = np.random.uniform(-1, 1, size=2)
        y_target = x[0] + x[1]
        y_pred = agent.act(x)
        agent.observe(x, y_target)
        if step % 20 == 0:
            print(f"Step {step}: pred={y_pred:.3f}, target={y_target:.3f}")
    # Final test
    x_test = np.array([0.5, -0.2])
    print(f"Test input {x_test}, pred={agent.act(x_test):.3f}, target={x_test.sum():.3f}")
