"""
agent.py

Defines a generic Agent class for single and multiagent scenarios.
Supports integration with neuro-fuzzy models, transfer learning, and various environments.
"""

import numpy as np
from .neuro_fuzzy import NeuroFuzzyHybrid
from laws import enforce_laws

class Agent:
    """
    Generic agent that interacts with an environment using a model and policy.
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

class NeuroFuzzyAgent(Agent):
    """
    Agent that uses a NeuroFuzzyHybrid model to select actions.
    """
    def __init__(self, nn_config, fis_config, policy=None):
        model = NeuroFuzzyHybrid(nn_config, fis_config)
        if policy is None:
            policy = lambda obs, model: model.forward(obs)
        super().__init__(model, policy)

    def self_organize(self, *args, **kwargs):
        """
        Triggers self-organization in the underlying neuro-fuzzy model.
        """
        if hasattr(self.model, 'self_organize'):
            self.model.self_organize(*args, **kwargs)

class Agent:
    """
    Generic agent that interacts with an environment using a model and policy.
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
        # Enforce laws before returning the action
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

class NeuroFuzzyAgent(Agent):
    """
    Agent that uses a NeuroFuzzyHybrid model to select actions.
    """
    def __init__(self, nn_config, fis_config, policy=None):
        model = NeuroFuzzyHybrid(nn_config, fis_config)
        if policy is None:
            policy = lambda obs, model: model.forward(obs)
        super().__init__(model, policy)

    def self_organize(self, *args, **kwargs):
        """
        Triggers self-organization in the underlying neuro-fuzzy model.
        """
        if hasattr(self.model, 'self_organize'):
            self.model.self_organize(*args, **kwargs)
