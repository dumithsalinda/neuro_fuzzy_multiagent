from .base_env import BaseEnvironment
import numpy as np

class MultiAgentGridworldEnv(BaseEnvironment):
    """
    Example refactored environment: Multi-Agent Gridworld
    Now inherits from BaseEnvironment and implements required methods.
    """
    def __init__(self, grid_size=5, n_agents=3, n_obstacles=2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_obstacles = n_obstacles
        self.reset()

    def reset(self):
        # Initialize agent and obstacle positions
        self.agent_positions = [self._random_pos() for _ in range(self.n_agents)]
        self.obstacle_positions = [self._random_pos() for _ in range(self.n_obstacles)]
        self.timestep = 0
        return self.get_observation()

    def step(self, actions):
        # Apply actions (list of actions for each agent)
        rewards = []
        for i, action in enumerate(actions):
            self.agent_positions[i] = self._move(self.agent_positions[i], action)
            rewards.append(-1)  # Example: -1 per step
        self.timestep += 1
        obs = self.get_observation()
        done = self.timestep >= 100
        info = {}
        return obs, rewards, done, info

    def render(self, mode="human"):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for pos in self.agent_positions:
            grid[pos[0], pos[1]] = 1
        for pos in self.obstacle_positions:
            grid[pos[0], pos[1]] = 2
        print(grid)

    def get_observation(self):
        # Return agent-agnostic observation (positions as flat array)
        obs = np.concatenate([
            np.array(self.agent_positions).flatten(),
            np.array(self.obstacle_positions).flatten()
        ])
        return obs

    def get_state(self):
        # Return full state for logging/advanced agents
        return {
            "agent_positions": self.agent_positions,
            "obstacle_positions": self.obstacle_positions,
            "timestep": self.timestep
        }

    @property
    def action_space(self):
        # 0: stay, 1: up, 2: down, 3: left, 4: right
        return 5

    @property
    def observation_space(self):
        # Flat array: [agents * 2 + obstacles * 2]
        return self.n_agents * 2 + self.n_obstacles * 2

    def _random_pos(self):
        return [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]

    def _move(self, pos, action):
        # Simple move logic
        x, y = pos
        if action == 1 and x > 0: x -= 1
        elif action == 2 and x < self.grid_size - 1: x += 1
        elif action == 3 and y > 0: y -= 1
        elif action == 4 and y < self.grid_size - 1: y += 1
        return [x, y]
