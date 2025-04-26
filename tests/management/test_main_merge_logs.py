from dashboard.main import merge_logs
from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.agents.neuro_fuzzy_fusion_agent import (
    NeuroFuzzyFusionAgent,
)


def test_merge_logs_merges_and_sorts():
    local_log = [
        {"time": "2024-01-01T10:00:00", "agent": 1, "group": 0, "user": "a"},
        {"time": "2024-01-01T10:01:00", "agent": 2, "group": 0, "user": "b"},
    ]
    remote_log = [
        {"time": "2024-01-01T10:00:00", "agent": 1, "group": 0, "user": "a"},
        {"time": "2024-01-01T09:59:00", "agent": 3, "group": 1, "user": "c"},
    ]
    merged = merge_logs(local_log, remote_log)
    assert len(merged) == 3
    assert merged[0]["time"] == "2024-01-01T09:59:00"
    assert merged[-1]["time"] == "2024-01-01T10:01:00"
