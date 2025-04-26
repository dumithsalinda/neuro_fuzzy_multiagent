"""
test_distributed_agent_executor.py

Test for distributed agent execution utility using Ray.
"""

import numpy as np

from neuro_fuzzy_multiagent.core.management.distributed_agent_executor import (
    run_agents_distributed,
)


class DummyAgent:
    def __init__(self, idx):
        self.idx = idx

    def act(self, obs):
        # Simulate computation
        return obs * (self.idx + 1)


def test_run_agents_distributed():
    agents = [DummyAgent(i) for i in range(4)]
    observations = np.arange(1, 5)  # [1, 2, 3, 4]
    actions = run_agents_distributed(agents, observations)
    # Each action should be obs * (agent idx + 1)
    expected = [obs * (i + 1) for i, obs in enumerate(observations)]
    assert actions == expected
    print("Distributed actions:", actions)
