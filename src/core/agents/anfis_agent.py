import numpy as np

from src.core.plugins.registration_utils import register_plugin
from src.core.agents.anfis_hybrid import ANFISHybrid


@register_plugin("agent")
class NeuroFuzzyANFISAgent:
    """
    Agent wrapper for the ANFISHybrid neuro-fuzzy model.
    Supports act (forward), observe (online update), experience replay for continual learning,
    and meta-learning hooks (e.g., adaptive learning rate, learning-to-learn).

    meta_update_fn: Optional callback called after each update.
      Signature: fn(agent, step: int) -> None
      agent: the NeuroFuzzyANFISAgent instance
      step: current update step
    """

    def __init__(
        self,
        input_dim,
        n_rules,
        lr=0.01,
        buffer_size=100,
        replay_enabled=True,
        replay_batch=8,
        meta_update_fn=None,
    ):
        self.model = ANFISHybrid(input_dim, n_rules)
        self.lr = lr
        self.last_obs = None
        self.last_action = None
        # Experience replay
        self.buffer_size = buffer_size
        self.replay_enabled = replay_enabled
        self.replay_batch = replay_batch
        self.replay_buffer = []
        # Meta-learning
        self.meta_update_fn = meta_update_fn
        self.update_step = 0

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
        # Meta-learning hook
        self.update_step += 1
        if self.meta_update_fn is not None:
            self.meta_update_fn(self, self.update_step)

    def replay_sample(self):
        idxs = np.random.choice(
            len(self.replay_buffer), self.replay_batch, replace=False
        )
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
    print(
        f"Test input {x_test}, pred={agent.act(x_test):.3f}, target={x_test.sum():.3f}"
    )
