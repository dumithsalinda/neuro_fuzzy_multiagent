"""
CurriculumScenarioGenerator: yields scenarios of increasing difficulty for curriculum learning.
"""

import itertools


class CurriculumScenarioGenerator:
    def __init__(self, base_grid, curriculum_steps):
        """
        base_grid: dict, parameter grid for scenarios (e.g., agent_type, env, seed)
        curriculum_steps: list of dicts, each dict contains parameters to increase difficulty (e.g., agent_count, obstacle_count)
        """
        self.base_grid = base_grid
        self.curriculum_steps = curriculum_steps

    def curriculum(self):
        """
        Yields scenarios for each curriculum stage, increasing difficulty.
        """
        for step_params in self.curriculum_steps:
            # Merge step_params into base_grid
            merged_grid = {**self.base_grid, **step_params}
            keys = list(merged_grid.keys())
            values = [
                merged_grid[k] if isinstance(merged_grid[k], list) else [merged_grid[k]]
                for k in keys
            ]
            for combo in itertools.product(*values):
                yield dict(zip(keys, combo))
