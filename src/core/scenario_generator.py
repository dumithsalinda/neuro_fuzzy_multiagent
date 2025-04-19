import itertools
import random

class ScenarioGenerator:
    """
    Utility to generate experiment scenarios/curricula for benchmarking.
    Supports grid search, random sampling, and curriculum generation.
    """
    def __init__(self, param_grid: dict):
        self.param_grid = param_grid

    def grid_search(self):
        # Returns a list of dicts (all combinations)
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combos = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combos]

    def random_sample(self, n=10):
        # Returns n random scenarios
        keys = list(self.param_grid.keys())
        scenarios = []
        for _ in range(n):
            scenario = {k: random.choice(self.param_grid[k]) for k in keys}
            scenarios.append(scenario)
        return scenarios

    def curriculum(self, order_by: str):
        # Returns scenarios sorted by a parameter (e.g., increasing agent count)
        scenarios = self.grid_search()
        return sorted(scenarios, key=lambda x: x[order_by])

# Usage Example:
# grid = {"agent_type": ["DQNAgent", "NeuroFuzzyAgent"], "agent_count": [2, 4, 8], "env": ["Gridworld", "IoTEnv"]}
# gen = ScenarioGenerator(grid)
# for scenario in gen.grid_search():
#     print(scenario)
