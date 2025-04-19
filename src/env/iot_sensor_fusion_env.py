import numpy as np
from .base_env import BaseEnvironment

class IoTSensorFusionEnv(BaseEnvironment):
    """
    Hybrid environment: gridworld with simulated IoT sensors (can be extended to real sensors).
    State = [agent positions, temperature, humidity, light].
    Optional: replace sensor simulation with real-world data.
    """
    def __init__(self, grid_size=5, n_agents=2, use_real_sensors=False):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.use_real_sensors = use_real_sensors
        self.reset()

    def reset(self):
        self.agent_positions = [self._random_pos() for _ in range(self.n_agents)]
        self.timestep = 0
        self._update_sensors()
        return self.get_observation()

    def step(self, actions):
        # Actions: 0=up, 1=down, 2=left, 3=right
        for i, action in enumerate(actions):
            self.agent_positions[i] = self._move(self.agent_positions[i], action)
        self.timestep += 1
        self._update_sensors()
        obs = self.get_observation()
        reward = self._compute_reward()
        done = self.timestep >= 100
        return obs, reward, done, {}

    def render(self, mode="human"):
        print(f"Agents: {self.agent_positions}, Sensors: T={self.temperature}, H={self.humidity}, L={self.light}")

    def get_observation(self):
        return {
            "agent_positions": [pos.copy() for pos in self.agent_positions],
            "temperature": self.temperature,
            "humidity": self.humidity,
            "light": self.light
        }

    def get_state(self):
        return {
            "agent_positions": [pos.copy() for pos in self.agent_positions],
            "temperature": self.temperature,
            "humidity": self.humidity,
            "light": self.light,
            "timestep": self.timestep
        }

    @property
    def action_space(self):
        return 4  # up, down, left, right

    @property
    def observation_space(self):
        return self.n_agents * 2 + 3  # agent positions + 3 sensors

    def _random_pos(self):
        return [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]

    def _move(self, pos, action):
        x, y = pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.grid_size - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.grid_size - 1: x += 1
        return [x, y]

    def _update_sensors(self):
        if self.use_real_sensors:
            # Placeholder for real sensor integration
            self.temperature = self._read_real_sensor("temperature")
            self.humidity = self._read_real_sensor("humidity")
            self.light = self._read_real_sensor("light")
        else:
            # Simulate sensor values
            self.temperature = 20 + 5 * np.sin(self.timestep / 10)
            self.humidity = 50 + 10 * np.cos(self.timestep / 15)
            self.light = 100 + 20 * np.sin(self.timestep / 5)

    def _read_real_sensor(self, sensor_type):
        # Stub for real sensor reading
        return 0.0

