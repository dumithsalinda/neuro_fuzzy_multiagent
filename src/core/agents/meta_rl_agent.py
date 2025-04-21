import numpy as np
from src.core.agents.agent import Agent

class MetaRLAgent(Agent):
    """
    Meta-Reinforcement Learning Agent.
    This agent adapts its own learning strategy (e.g., learning rate, exploration policy, even algorithm) based on meta-feedback about its performance.
    """
    def __init__(self, base_agent_cls, base_agent_kwargs, meta_lr=0.1, meta_window=10):
        super().__init__(model=None)
        self.base_agent_cls = base_agent_cls
        self.base_agent_kwargs = base_agent_kwargs.copy()
        self.meta_lr = meta_lr
        self.meta_window = meta_window
        self.base_agent = self.base_agent_cls(**self.base_agent_kwargs)
        self.reward_history = []
        self.meta_state = {'lr': self.base_agent_kwargs.get('lr', 0.1)}

    def act(self, observation, state=None):
        return self.base_agent.act(observation, state)

    def observe(self, reward, next_state, done):
        self.base_agent.observe(reward, next_state, done)
        self.reward_history.append(reward)
        if len(self.reward_history) >= self.meta_window:
            self.meta_update()
            self.reward_history = []

    def meta_update(self):
        # Example: meta-learn the learning rate based on average reward improvement
        avg_reward = np.mean(self.reward_history)
        if hasattr(self.base_agent, 'lr'):
            # Increase lr if reward is improving, decrease if not
            if avg_reward > 0:
                self.meta_state['lr'] = min(self.meta_state['lr'] + self.meta_lr, 1.0)
            else:
                self.meta_state['lr'] = max(self.meta_state['lr'] - self.meta_lr, 0.0001)
            self.base_agent.lr = self.meta_state['lr']

    def reset(self):
        self.base_agent = self.base_agent_cls(**self.base_agent_kwargs)
        self.reward_history = []
        self.meta_state = {'lr': self.base_agent_kwargs.get('lr', 0.1)}
