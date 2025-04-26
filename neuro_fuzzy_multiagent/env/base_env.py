from abc import ABC, abstractmethod
from typing import Any

from neuro_fuzzy_multiagent.core.plugins.registration_utils import register_plugin


@register_plugin("environment")
class BaseEnvironment(ABC):
    """
    Abstract base class for all environments (simulated, real-world, etc.).

    Defines the interface that all environments must implement, including reset, step, get_state, get_observation, and render.
    Provides hooks for real-world integration (API, sensor, robot) and real-time data injection via set_external_input.
    Use set_external_input(agent_idx: int, value: Any) to inject live values (from REST, MQTT, sensors, etc.) into the environment for agent control or observation override.
    """

    def set_external_input(self, agent_idx: int, value: Any) -> None:
        """
        Inject a real-time value for the given agent index.
        Override in subclasses to use value in observation/state.
        Args:
            agent_idx (int): Index of the agent to inject the value for.
            value (Any): The value to inject (float, dict, etc.).
        """
        if not hasattr(self, "_external_inputs"):
            self._external_inputs = {}
        self._external_inputs[agent_idx] = value

    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to an initial state. Returns initial observation/state."""

    @abstractmethod
    def step(self, action):
        """Apply an action, return (observation, reward, done, info)."""

    @abstractmethod
    def render(self, mode="human"):
        """Render the environment (optional for headless environments)."""

    @abstractmethod
    def get_observation(self):
        """Return the current observation (agent-agnostic format preferred)."""

    @abstractmethod
    def get_state(self):
        """Return the full environment state (for logging or advanced agents)."""

    # --- Real-world integration hooks (optional) ---
    def connect(self):
        """Connect to real-world API, robot, or sensor (if applicable)."""
        raise NotImplementedError("connect() not implemented for this environment.")

    def disconnect(self):
        """Disconnect from hardware/API (if applicable)."""
        raise NotImplementedError("disconnect() not implemented for this environment.")

    def send_action_to_hardware(self, action):
        """Send action to robot or real-world actuator (if applicable)."""
        raise NotImplementedError(
            "send_action_to_hardware() not implemented for this environment."
        )

    def read_sensor_data(self):
        """Read sensor data from hardware/API (if applicable)."""
        raise NotImplementedError(
            "read_sensor_data() not implemented for this environment."
        )

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space object or description."""

    @property
    @abstractmethod
    def observation_space(self):
        """Return the observation space object or description."""

    def perceive(self):
        """Return the current observation/state for the agent (default: get_observation)."""
        return self.get_observation()

    def extract_features(self, state=None):
        """Convert raw state to features for learning or transfer (default: identity mapping)."""
        import numpy as np

        if state is None:
            state = self.get_observation()
        return np.array(state)

    """
    Abstract base class for all environments (simulated, real-world, etc.)
    Defines the interface that all environments must implement.
    Optional hooks for real-world integration (API, sensor, robot) are provided.
    """

    @abstractmethod
    def reset(self):
        """Reset the environment to an initial state. Returns initial observation/state."""

    @abstractmethod
    def step(self, action):
        """Apply an action, return (observation, reward, done, info)."""

    @abstractmethod
    def render(self, mode="human"):
        """Render the environment (optional for headless environments)."""

    @abstractmethod
    def get_observation(self):
        """Return the current observation (agent-agnostic format preferred)."""

    @abstractmethod
    def get_state(self):
        """Return the full environment state (for logging or advanced agents)."""

    # --- Real-world integration hooks (optional) ---
    def connect(self):
        """Connect to real-world API, robot, or sensor (if applicable)."""
        raise NotImplementedError("connect() not implemented for this environment.")

    def disconnect(self):
        """Disconnect from hardware/API (if applicable)."""
        raise NotImplementedError("disconnect() not implemented for this environment.")

    def send_action_to_hardware(self, action):
        """Send action to robot or real-world actuator (if applicable)."""
        raise NotImplementedError(
            "send_action_to_hardware() not implemented for this environment."
        )

    def read_sensor_data(self):
        """Read sensor data from hardware/API (if applicable)."""
        raise NotImplementedError(
            "read_sensor_data() not implemented for this environment."
        )

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space object or description."""

    @property
    @abstractmethod
    def observation_space(self):
        """Return the observation space object or description."""
