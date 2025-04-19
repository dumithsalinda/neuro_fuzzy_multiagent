"""
abstraction.py

Defines core environment abstractions for neuro-fuzzy multiagent systems.
Includes:
- NoisyEnvironment: Adds Gaussian noise for transfer learning/domain adaptation.
- SimpleEnvironment: Minimal environment for basic testing and demonstration.

These abstractions support transfer learning, feature extraction, and robust agent evaluation across domains.
"""

import numpy as np

from src.env.base_env import BaseEnvironment


class NoisyEnvironment(BaseEnvironment):
    """
    Environment with a random state vector plus Gaussian noise.
    Useful for simulating domain shift and testing transfer learning robustness.
    """

    def __init__(self, dim=3, noise_std=0.5):
        self.dim = dim
        self.noise_std = noise_std
        self.state = np.zeros(dim)

    def reset(self):
        self.state = np.random.randn(self.dim) + np.random.normal(
            0, self.noise_std, self.dim
        )
        return self.get_observation()

    def step(self, action):
        self.state += action + np.random.normal(0, self.noise_std, self.dim)
        return self.get_observation(), 0.0, False, {}

    def render(self, mode="human"):
        print(f"NoisyEnvironment State: {self.state}")

    def get_observation(self):
        return self.state.copy()

    def get_state(self):
        return self.state.copy()

    @property
    def action_space(self):
        return self.dim

    @property
    def observation_space(self):
        return self.dim

    def perceive(self):
        return self.get_observation()

    def extract_features(self, state=None):
        if state is None:
            state = self.state
        return np.array(state)


class SimpleEnvironment(BaseEnvironment):
    """
    A simple toy environment with a random state vector.
    Useful for basic testing or as a source domain for transfer learning.
    """

    def __init__(self, dim=3):
        self.dim = dim
        self.state = np.zeros(dim)

    def reset(self):
        self.state = np.random.randn(self.dim)
        return self.get_observation()

    def step(self, action):
        self.state += action
        return self.get_observation(), 0.0, False, {}

    def render(self, mode="human"):
        print(f"SimpleEnvironment State: {self.state}")

    def get_observation(self):
        return self.state.copy()

    def get_state(self):
        return self.state.copy()

    @property
    def action_space(self):
        return self.dim

    @property
    def observation_space(self):
        return self.dim

    def perceive(self):
        return self.get_observation()

    def extract_features(self, state=None):
        if state is None:
            state = self.state
        return np.array(state)

    """
    Environment with a random state vector plus Gaussian noise.
    Useful for simulating domain shift and testing transfer learning robustness.

    Parameters
    ----------
    dim : int
        Dimensionality of the state vector.
    noise_std : float
        Standard deviation of the Gaussian noise added to the state.
    """

    def __init__(self, dim=3, noise_std=0.5):
        """
        Initialize a NoisyEnvironment.

        Parameters
        ----------
        dim : int
            State dimensionality.
        noise_std : float
            Standard deviation for Gaussian noise.
        """
        self.dim = dim
        self.noise_std = noise_std
        self.state = np.zeros(dim)

    def reset(self):
        """
        Reset the environment state with Gaussian noise.

        Returns
        -------
        np.ndarray
            The new noisy state vector.
        """
        self.state = np.random.randn(self.dim) + np.random.normal(
            0, self.noise_std, self.dim
        )
        return self.state

    def step(self, action):
        """
        Apply an action and add Gaussian noise.

        Parameters
        ----------
        action : np.ndarray
            Action vector to apply.
        Returns
        -------
        np.ndarray
            Updated noisy state vector.
        """
        self.state += action + np.random.normal(0, self.noise_std, self.dim)
        return self.state

    def perceive(self):
        """Return the current (noisy) state."""
        return self.state

    def extract_features(self, state=None):
        """
        Extract features from the state (identity mapping).

        Parameters
        ----------
        state : np.ndarray or None
            State to extract features from. Defaults to current state.
        Returns
        -------
        np.ndarray
            Feature vector.
        """
        if state is None:
            state = self.state
        return np.array(state)


class SimpleEnvironment(BaseEnvironment):
    """
    A simple toy environment with a random state vector.
    Useful for basic testing or as a source domain for transfer learning.

    Parameters
    ----------
    dim : int
        Dimensionality of the state vector.
    """

    def __init__(self, dim=3):
        """
        Initialize a SimpleEnvironment.

        Parameters
        ----------
        dim : int
            State dimensionality.
        """
        self.dim = dim
        self.state = np.zeros(dim)

    def reset(self):
        """
        Reset the environment state to a random vector.
        Returns
        -------
        np.ndarray
            The new state vector.
        """
        self.state = np.random.randn(self.dim)
        return self.get_observation()

    def step(self, action):
        """
        Apply an action to the state.
        Parameters
        ----------
        action : np.ndarray
            Action vector to apply.
        Returns
        -------
        tuple
            (observation, reward, done, info)
        """
        self.state += action
        return self.get_observation(), 0.0, False, {}

    def render(self, mode="human"):
        print(f"SimpleEnvironment State: {self.state}")

    def get_observation(self):
        return self.state.copy()

    def get_state(self):
        return self.state.copy()

    @property
    def action_space(self):
        return self.dim

    @property
    def observation_space(self):
        return self.dim

    def perceive(self):
        return self.get_observation()

    def extract_features(self, state=None):
        if state is None:
            state = self.state
        return np.array(state)
