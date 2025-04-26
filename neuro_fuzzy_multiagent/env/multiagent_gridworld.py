import numpy as np

from neuro_fuzzy_multiagent.env.base_env import BaseEnvironment


class MultiAgentGridworldEnv(BaseEnvironment):
    """
    Multi-agent gridworld environment with optional obstacles.
    Agents move on a grid, can be cooperative or competitive.
    """

    def __init__(self, grid_size=5, n_agents=2, mode="cooperative", n_obstacles=0):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.mode = mode  # "cooperative" or "competitive"
        self.n_obstacles = n_obstacles
        self.reset()

    def reset(self):
        # Place agents and target first
        taken = set()
        self.agent_positions = []
        for _ in range(self.n_agents):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if pos not in taken:
                    self.agent_positions.append(pos)
                    taken.add(pos)
                    break
        while True:
            self.target = tuple(np.random.randint(0, self.grid_size, size=2))
            if self.target not in taken:
                taken.add(self.target)
                break
        # Place obstacles
        self.obstacles = []
        for _ in range(self.n_obstacles):
            while True:
                obs = tuple(np.random.randint(0, self.grid_size, size=2))
                if obs not in taken:
                    self.obstacles.append(obs)
                    taken.add(obs)
                    break
        self.done = [False] * self.n_agents
        return self.get_observation()

    def _get_obs(self):
        # Each agent observes its own position, the target, and (optionally) obstacles
        # For now, obstacles are not included in obs vector for simplicity
        return [
            np.array(list(self.agent_positions[i]) + list(self.target))
            for i in range(self.n_agents)
        ]

    def get_observation(self):
        return self._get_obs()

    def get_state(self):
        return {
            "agent_positions": self.agent_positions,
            "target": self.target,
            "obstacles": self.obstacles,
            "done": self.done,
        }

    def render(self, mode="human"):
        print(
            f"Agents: {self.agent_positions}, Target: {self.target}, Obstacles: {self.obstacles}"
        )

    @property
    def action_space(self):
        return 4  # up, down, left, right

    @property
    def observation_space(self):
        # own pos (2) + target (2)
        return 4

    def step(self, actions):
        rewards = [0] * self.n_agents
        for i, action in enumerate(actions):
            if self.done[i]:
                continue
            # Actions: 0=up, 1=down, 2=left, 3=right
            x, y = self.agent_positions[i]
            nx, ny = x, y
            if action == 0 and y > 0:
                ny -= 1
            elif action == 1 and y < self.grid_size - 1:
                ny += 1
            elif action == 2 and x > 0:
                nx -= 1
            elif action == 3 and x < self.grid_size - 1:
                nx += 1
            # Check if new position is an obstacle
            if (nx, ny) not in self.obstacles:
                self.agent_positions[i] = (nx, ny)
            # Check if reached target
            if self.agent_positions[i] == self.target:
                self.done[i] = True
                if self.mode == "cooperative":
                    rewards[i] = 1
                elif self.mode == "competitive":
                    rewards[i] = 1
                    # Only first agent to reach gets reward, others get 0
                    for j in range(self.n_agents):
                        if j != i:
                            self.done[j] = True
        obs = self._get_obs()
        done = all(self.done)
        return obs, rewards, done
