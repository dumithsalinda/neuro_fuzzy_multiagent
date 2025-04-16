import numpy as np

class MultiAgentResourceEnv:
    """
    Multi-agent resource collection environment (grid).
    Agents collect resources for reward. Supports cooperative, competitive, or mixed.
    """
    def __init__(self, grid_size=5, n_agents=2, n_resources=3, mode="competitive"):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.mode = mode  # "cooperative", "competitive", or "mixed"
        self.reset()

    def reset(self):
        self.agent_positions = [tuple(np.random.randint(0, self.grid_size, size=2)) for _ in range(self.n_agents)]
        self.resource_positions = set()
        while len(self.resource_positions) < self.n_resources:
            self.resource_positions.add(tuple(np.random.randint(0, self.grid_size, size=2)))
        self.resource_positions = set(self.resource_positions)
        self.collected = [0] * self.n_agents
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        # Each agent observes its own position and all resource positions
        return [np.array(list(self.agent_positions[i]) + [coord for pos in self.resource_positions for coord in pos]) for i in range(self.n_agents)]

    def step(self, actions):
        rewards = [0] * self.n_agents
        for i, action in enumerate(actions):
            # Actions: 0=up, 1=down, 2=left, 3=right
            x, y = self.agent_positions[i]
            if action == 0 and y > 0:
                y -= 1
            elif action == 1 and y < self.grid_size - 1:
                y += 1
            elif action == 2 and x > 0:
                x -= 1
            elif action == 3 and x < self.grid_size - 1:
                x += 1
            self.agent_positions[i] = (x, y)
        # Check for resource collection
        for i, pos in enumerate(self.agent_positions):
            if pos in self.resource_positions:
                self.resource_positions.remove(pos)
                self.collected[i] += 1
                if self.mode == "cooperative":
                    for j in range(self.n_agents):
                        rewards[j] += 1
                elif self.mode == "competitive":
                    rewards[i] += 1
                elif self.mode == "mixed":
                    rewards[i] += 1  # customize as needed
        if not self.resource_positions:
            self.done = True
        obs = self._get_obs()
        return obs, rewards, self.done
