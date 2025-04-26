import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
from neuro_fuzzy_multiagent.dashboard.visualization import (
    render_agent_positions,
    render_group_knowledge,
    render_group_analytics,
)


def test_render_agent_positions_runs():
    agents = [MagicMock(group=0), MagicMock(group=1)]
    positions = [(0, 0), (1, 1)]
    with patch("dashboard.visualization.st", st):
        render_agent_positions(positions, agents)


def test_render_group_knowledge_runs():
    class DummyMAS:
        def __init__(self):
            self.groups = {0: [0, 1]}
            self.group_leaders = {0: 0}
            self.agents = [
                MagicMock(share_knowledge=lambda: {"a": 1}),
                MagicMock(share_knowledge=lambda: {"b": 2}),
            ]

    mas = DummyMAS()
    with patch("dashboard.visualization.st", st):
        render_group_knowledge(mas)


def test_render_group_analytics_runs():
    class DummyMAS:
        def __init__(self):
            self.groups = {0: [0, 1]}
            self.agents = [MagicMock(), MagicMock()]

    mas = DummyMAS()
    reward_history = {0: [1, 2], 1: [3, 4]}
    with patch("dashboard.visualization.st", st):
        render_group_analytics(mas, reward_history)
