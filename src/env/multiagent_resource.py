import numpy as np

from .base_env import BaseEnvironment

class MultiAgentResourceEnv(BaseEnvironment):
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
        return self.get_observation()

    def _get_obs(self):
        # Each agent observes its own position and all resource positions
        return [np.array(list(self.agent_positions[i]) + [coord for pos in self.resource_positions for coord in pos]) for i in range(self.n_agents)]

    def get_observation(self):
        return self._get_obs()

    def perceive(self):
        return self.get_observation()

    def extract_features(self, state=None):
        import numpy as np
        if state is None:
            state = self.get_observation()
        # state is a list of arrays (one per agent)
        return [np.array(s) for s in state]


    def get_state(self):
        return {
            "agent_positions": self.agent_positions,
            "resource_positions": list(self.resource_positions),
            "collected": self.collected,
            "done": self.done
        }

    def render(self, mode="human"):
        print(f"Agents: {self.agent_positions}, Resources: {self.resource_positions}, Collected: {self.collected}")

    @property
    def action_space(self):
        return 4  # up, down, left, right

    @property
    def observation_space(self):
        # own pos (2) + all resources (2 * n_resources)
        return 2 + 2 * self.n_resources

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
