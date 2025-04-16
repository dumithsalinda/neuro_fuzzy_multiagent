"""
abstraction.py

Defines core environment abstractions for neuro-fuzzy multiagent systems.
Includes:
- Environment: Abstract base class for all environments.
- NoisyEnvironment: Adds Gaussian noise for transfer learning/domain adaptation.
- SimpleEnvironment: Minimal environment for basic testing and demonstration.

These abstractions support transfer learning, feature extraction, and robust agent evaluation across domains.
"""

import numpy as np

class Environment:
    """
    Abstract environment interface for perception and feature extraction.

    Methods
    -------
    reset():
        Reset the environment state to a starting configuration.
    step(action):
        Apply an action and update the environment state.
    perceive():
        Return the current observation/state.
    extract_features(state=None):
        Convert raw state to features for learning or transfer.
    """
    def reset(self):
        """Reset the environment state to a starting configuration."""
        raise NotImplementedError
    def step(self, action):
        """Apply an action and update the environment state."""
        raise NotImplementedError
    def perceive(self):
        """Return the current observation/state."""
        raise NotImplementedError
    def extract_features(self, state=None):
        """Convert raw state to features for learning or transfer."""
        raise NotImplementedError

class NoisyEnvironment(Environment):
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
        self.state = np.random.randn(self.dim) + np.random.normal(0, self.noise_std, self.dim)
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

class SimpleEnvironment(Environment):
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
        return self.state
    def step(self, action):
        """
        Apply an action to the state.

        Parameters
        ----------
        action : np.ndarray
            Action vector to apply.
        Returns
        -------
        np.ndarray
            Updated state vector.
        """
        self.state += action
        return self.state
    def perceive(self):
        """Return the current state."""
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
