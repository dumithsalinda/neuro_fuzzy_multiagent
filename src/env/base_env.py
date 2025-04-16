from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Abstract base class for all environments (simulated, real-world, etc.)
    Defines the interface that all environments must implement.
    """

    @abstractmethod
    def reset(self):
        """Reset the environment to an initial state. Returns initial observation/state."""
        pass

    @abstractmethod
    def step(self, action):
        """Apply an action, return (observation, reward, done, info)."""
        pass

    @abstractmethod
    def render(self, mode="human"):
        """Render the environment (optional for headless environments)."""
        pass

    @abstractmethod
    def get_observation(self):
        """Return the current observation (agent-agnostic format preferred)."""
        pass

    @abstractmethod
    def get_state(self):
        """Return the full environment state (for logging or advanced agents)."""
        pass

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space object or description."""
        pass

    @property
    @abstractmethod
    def observation_space(self):
        """Return the observation space object or description."""
        pass
