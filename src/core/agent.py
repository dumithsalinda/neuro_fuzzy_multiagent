"""
agent.py

Defines a generic Agent class for single and multiagent scenarios.
Supports integration with neuro-fuzzy models, transfer learning, and various environments.
"""

import numpy as np
from .neuro_fuzzy import NeuroFuzzyHybrid
from laws import enforce_laws
from .online_learning import OnlineLearnerMixin

class Agent(OnlineLearnerMixin):
    """
    Generic agent that interacts with an environment using a model and policy.
    Now supports online learning from web resources.
    """
    def __init__(self, model, policy=None):
        self.model = model
        self.policy = policy if policy is not None else self.random_policy
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0

    def act(self, observation, state=None):
        """
        Select an action based on the current observation, enforcing unbreakable laws.
        """
        action = self.policy(observation, self.model)
        enforce_laws(action, state if state is not None else {})
        self.last_action = action
        self.last_observation = observation
        return action

    def observe(self, reward):
        """
        Receive reward and update internal state.
        """
        self.total_reward += reward

    def learn(self, *args, **kwargs):
        """
        Placeholder for agent learning (to be overridden for RL, etc.).
        """
        pass

    def self_organize(self, *args, **kwargs):
        """
        Placeholder for agent self-organization (to be overridden).
        """
        pass

    def reset(self):
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0

    @staticmethod
    def random_policy(observation, model):
        # For demonstration: random action in the same shape as observation
        return np.random.randn(*observation.shape)

    def integrate_online_knowledge(self, knowledge):
        """
        Default: try to update policy/model if possible, else store as attribute.
        """
        if hasattr(self.model, 'update_from_knowledge'):
            self.model.update_from_knowledge(knowledge)
        else:
            self.online_knowledge = knowledge

class NeuroFuzzyAgent(Agent):
    """
    Agent that uses a NeuroFuzzyHybrid model to select actions.
    Supports modular self-organization of fuzzy rules, membership functions, and neural network weights.
    """
    def __init__(self, nn_config, fis_config, policy=None):
        model = NeuroFuzzyHybrid(nn_config, fis_config)
        if policy is None:
            policy = lambda obs, model: model.forward(obs)
        super().__init__(model, policy)

    def self_organize(self, mutation_rate=0.01, tune_fuzzy=True, rule_change=True):
        """
        Trigger self-organization in the underlying neuro-fuzzy hybrid model.
        This adapts neural weights, tunes fuzzy sets, and adds/removes rules.
        """
        if hasattr(self.model, 'self_organize'):
            self.model.self_organize(mutation_rate=mutation_rate, tune_fuzzy=tune_fuzzy, rule_change=rule_change)
