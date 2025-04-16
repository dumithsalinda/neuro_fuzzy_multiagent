"""
agent.py

Defines a generic Agent class for single and multiagent scenarios.
Supports integration with neuro-fuzzy models, transfer learning, and various environments.
"""

import numpy as np

from .neuro_fuzzy import NeuroFuzzyHybrid

class Agent:
    """
    Generic agent that interacts with an environment using a model and policy.

    Parameters
    ----------
    model : object
        The agent's model (must implement a .forward() method).
    policy : callable, optional
        Function to select actions given state/observation.
    """
    def __init__(self, model, policy=None):
        self.model = model
        self.policy = policy if policy is not None else self.random_policy
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0

    def act(self, observation):
        """
        Select an action based on the current observation.
        """
        action = self.policy(observation, self.model)
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

    Parameters
    ----------
    nn_config : dict
        Neural network configuration for NeuroFuzzyHybrid.
    fis_config : dict
        Fuzzy inference system configuration for NeuroFuzzyHybrid.
    policy : callable, optional
        Policy function (defaults to greedy via model.forward).
    """
    def __init__(self, nn_config, fis_config, policy=None):
        model = NeuroFuzzyHybrid(nn_config, fis_config)
        if policy is None:
            # Greedy: action = model.forward(observation)
            policy = lambda obs, model: model.forward(obs)
        super().__init__(model, policy)
