import numpy as np
import pytest

from dashboard.simulation import run_batch_experiments
from src.core.management.multiagent import MultiAgentSystem


def test_run_batch_experiments_basic():
    results = run_batch_experiments(
        n_experiments=2,
        agent_counts_list=[5, 10],
        seeds_list=[1, 2],
        n_steps=5,
        fast_mode=True,
    )
    assert isinstance(results, list)
    assert len(results) == 4  # 2 agent counts x 2 seeds
    for res in results:
        assert "mean_reward" in res
        assert "diversity" in res
        assert "group_stability" in res
        assert "intervention_count" in res
        assert res["agent_count"] in [5, 10]
        assert res["seed"] in [1, 2]
        assert res["experiment"] > 0
        assert isinstance(res["mean_reward"], float)
        assert isinstance(res["diversity"], float)
        assert isinstance(res["group_stability"], float)
        assert isinstance(res["intervention_count"], int)
