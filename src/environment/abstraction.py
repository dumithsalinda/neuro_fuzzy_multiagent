import numpy as np

class Environment:
    """Abstract environment interface for perception and feature extraction."""
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
    """Environment with a random state vector plus Gaussian noise."""
    def __init__(self, dim=3, noise_std=0.5):
        self.dim = dim
        self.noise_std = noise_std
        self.state = np.zeros(dim)
    def reset(self):
        self.state = np.random.randn(self.dim) + np.random.normal(0, self.noise_std, self.dim)
        return self.state
    def step(self, action):
        self.state += action + np.random.normal(0, self.noise_std, self.dim)
        return self.state
    def perceive(self):
        return self.state
    def extract_features(self, state=None):
        if state is None:
            state = self.state
        return np.array(state)

class SimpleEnvironment(Environment):
    """A simple toy environment with a random state vector."""
    def __init__(self, dim=3):
        self.dim = dim
        self.state = np.zeros(dim)
    def reset(self):
        self.state = np.random.randn(self.dim)
        return self.state
    def step(self, action):
        self.state += action
        return self.state
    def perceive(self):
        return self.state
    def extract_features(self, state=None):
        if state is None:
            state = self.state
        return np.array(state)
