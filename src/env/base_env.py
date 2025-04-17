from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Abstract base class for all environments (simulated, real-world, etc.)
    Defines the interface that all environments must implement.
    Optional hooks for real-world integration (API, sensor, robot) are provided.
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

    # --- Real-world integration hooks (optional) ---
    def connect(self):
        """Connect to real-world API, robot, or sensor (if applicable)."""
        raise NotImplementedError("connect() not implemented for this environment.")

    def disconnect(self):
        """Disconnect from hardware/API (if applicable)."""
        raise NotImplementedError("disconnect() not implemented for this environment.")

    def send_action_to_hardware(self, action):
        """Send action to robot or real-world actuator (if applicable)."""
        raise NotImplementedError("send_action_to_hardware() not implemented for this environment.")

    def read_sensor_data(self):
        """Read sensor data from hardware/API (if applicable)."""
        raise NotImplementedError("read_sensor_data() not implemented for this environment.")

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
