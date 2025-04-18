import numpy as np

from .base_env import BaseEnvironment

class AdversarialGridworldEnv(BaseEnvironment):
    """
    Multi-agent gridworld with pursuers and evaders.
    Pursuers try to catch evaders; evaders try to reach target or avoid capture.
    """
    def __init__(self, grid_size=5, n_pursuers=1, n_evaders=1, n_obstacles=0):
        self.grid_size = grid_size
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_agents = n_pursuers + n_evaders
        self.n_obstacles = n_obstacles
        self.reset()

    def reset(self):
        taken = set()
        self.pursuer_positions = []
        self.evader_positions = []
        # Place pursuers
        for _ in range(self.n_pursuers):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if pos not in taken:
                    self.pursuer_positions.append(pos)
                    taken.add(pos)
                    break
        # Place evaders
        for _ in range(self.n_evaders):
            while True:
                pos = tuple(np.random.randint(0, self.grid_size, size=2))
                if pos not in taken:
                    self.evader_positions.append(pos)
                    taken.add(pos)
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
        # Place target for evaders
        while True:
            self.target = tuple(np.random.randint(0, self.grid_size, size=2))
            if self.target not in taken:
                taken.add(self.target)
                break
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        # Each agent observes its own position, all other agent positions, target, and obstacles
        # For simplicity, observation is a flat vector (all lists)
        pursuer_obs = [
            np.array(
                list(pos)
                + [coord for epos in self.evader_positions for coord in epos]
                + list(self.target)
                + [coord for obs in self.obstacles for coord in obs]
            )
            for pos in self.pursuer_positions
        ]
        evader_obs = [
            np.array(
                list(pos)
                + [coord for ppos in self.pursuer_positions for coord in ppos]
                + list(self.target)
                + [coord for obs in self.obstacles for coord in obs]
            )
            for pos in self.evader_positions
        ]
        return pursuer_obs + evader_obs

    def extract_features(self, state=None):
        import numpy as np
        if state is None:
            state = self.get_observation()
        # state is a list of arrays (one per agent)
        return [np.array(s) for s in state]


    def get_observation(self):
        return self._get_obs()

    def perceive(self):
        return self.get_observation()



    def get_state(self):
        return {
            "pursuer_positions": self.pursuer_positions,
            "evader_positions": self.evader_positions,
            "obstacles": self.obstacles,
            "target": self.target,
            "done": self.done
        }

    def render(self, mode="human"):
        print(f"Pursuers: {self.pursuer_positions}, Evaders: {self.evader_positions}, Target: {self.target}, Obstacles: {self.obstacles}")

    @property
    def action_space(self):
        return 5  # Example: 5 actions per agent

    @property
    def observation_space(self):
        # Example: own pos (2) + all other agents + target (2) + obstacles
        return 2 + 2 * (self.n_agents - 1) + 2 + 2 * self.n_obstacles

    def step(self, actions):
        # actions: list of ints, 0=up, 1=down, 2=left, 3=right, order: pursuers then evaders
        pursuer_actions = actions[:self.n_pursuers]
        evader_actions = actions[self.n_pursuers:]
        # Move pursuers
        for i, action in enumerate(pursuer_actions):
            x, y = self.pursuer_positions[i]
            nx, ny = x, y
            if action == 0 and y > 0:
                ny -= 1
            elif action == 1 and y < self.grid_size - 1:
                ny += 1
            elif action == 2 and x > 0:
                nx -= 1
            elif action == 3 and x < self.grid_size - 1:
                nx += 1
            if (nx, ny) not in self.obstacles:
                self.pursuer_positions[i] = (nx, ny)
        # Move evaders
        for i, action in enumerate(evader_actions):
            x, y = self.evader_positions[i]
            nx, ny = x, y
            if action == 0 and y > 0:
                ny -= 1
            elif action == 1 and y < self.grid_size - 1:
                ny += 1
            elif action == 2 and x > 0:
                nx -= 1
            elif action == 3 and x < self.grid_size - 1:
                nx += 1
            if (nx, ny) not in self.obstacles:
                self.evader_positions[i] = (nx, ny)
        # Check for capture or evader reaching target
        rewards = [0] * self.n_agents
        done = False
        # Check capture
        for i, ppos in enumerate(self.pursuer_positions):
            for j, epos in enumerate(self.evader_positions):
                if ppos == epos:
                    rewards[i] = 1   # pursuer gets reward
                    rewards[self.n_pursuers + j] = -1  # evader penalized
                    done = True
        # Check evader reaches target
        for j, epos in enumerate(self.evader_positions):
            if epos == self.target:
                rewards[self.n_pursuers + j] = 1
                done = True
        self.done = done
        return self._get_obs(), rewards, done
