import numpy as np

class MultiAgentGridworldEnv:
    """
    Multi-agent gridworld environment.
    Agents move on a grid, can be cooperative or competitive.
    """
    def __init__(self, grid_size=5, n_agents=2, mode="cooperative"):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.mode = mode  # "cooperative" or "competitive"
        self.reset()

    def reset(self):
        self.agent_positions = [tuple(np.random.randint(0, self.grid_size, size=2)) for _ in range(self.n_agents)]
        self.target = tuple(np.random.randint(0, self.grid_size, size=2))
        self.done = [False] * self.n_agents
        return self._get_obs()

    def _get_obs(self):
        # Each agent observes its own position and the target
        return [np.array(list(self.agent_positions[i]) + list(self.target)) for i in range(self.n_agents)]

    def step(self, actions):
        rewards = [0] * self.n_agents
        for i, action in enumerate(actions):
            if self.done[i]:
                continue
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
            # Check if reached target
            if (x, y) == self.target:
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
