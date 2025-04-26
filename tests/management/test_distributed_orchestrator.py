import pytest
from src.core.management.distributed_orchestrator import (
    DistributedExperimentOrchestrator,
)


class DummyAgent:
    def __init__(self):
        self.last_obs = None

    def act(self, obs, state=None):
        self.last_obs = obs
        return 1

    def observe(self, r, n, d):
        pass

    def reset(self):
        pass


class DummyEnv:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return 0

    def step(self, action):
        self.steps += 1
        return self.steps, 1, self.steps >= 5, {}


@pytest.mark.skipif("ray" not in globals(), reason="ray not installed")
def test_distributed_orchestrator():
    orchestrator = DistributedExperimentOrchestrator(DummyAgent, DummyEnv, num_agents=2)
    orchestrator.launch()
    results = orchestrator.run_episode(steps=5)
    assert len(results) == 2
    assert all(r == 5 for r in results)
    orchestrator.shutdown()
