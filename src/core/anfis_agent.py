import numpy as np
try:
    from .anfis_hybrid import ANFISHybrid
except ImportError:
    from anfis_hybrid import ANFISHybrid

class NeuroFuzzyANFISAgent:
    """
    Agent wrapper for the ANFISHybrid neuro-fuzzy model.
    Supports act (forward), observe (online update), and experience replay for continual learning.
    """
    def __init__(self, input_dim, n_rules, lr=0.01, buffer_size=100, replay_enabled=True, replay_batch=8):
        self.model = ANFISHybrid(input_dim, n_rules)
        self.lr = lr
        self.last_obs = None
        self.last_action = None
        # Experience replay
        self.buffer_size = buffer_size
        self.replay_enabled = replay_enabled
        self.replay_batch = replay_batch
        self.replay_buffer = []

    def act(self, obs):
        # Forward pass through ANFIS
        self.last_obs = obs
        action = self.model.forward(obs)
        self.last_action = action
        return action

    def observe(self, obs, reward):
        # Store experience
        self.replay_buffer.append((self.last_obs, reward))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
        # Online update: treat reward as target
        self.model.update(self.last_obs, reward, lr=self.lr)
        # Experience replay update
        if self.replay_enabled and len(self.replay_buffer) >= self.replay_batch:
            self.replay_update()

    def replay_sample(self):
        idxs = np.random.choice(len(self.replay_buffer), self.replay_batch, replace=False)
        return [self.replay_buffer[i] for i in idxs]

    def replay_update(self):
        batch = self.replay_sample()
        for obs, reward in batch:
            self.model.update(obs, reward, lr=self.lr)

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
