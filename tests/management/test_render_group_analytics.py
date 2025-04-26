from neuro_fuzzy_multiagent.dashboard.visualization import render_group_analytics
from unittest.mock import patch, MagicMock
import streamlit as st


def test_render_group_analytics_runs():
    class DummyMAS:
        def __init__(self):
            self.groups = {0: [0, 1]}
            self.agents = [MagicMock(), MagicMock()]

    mas = DummyMAS()
    reward_history = {0: [1, 2], 1: [3, 4]}
    with patch("dashboard.visualization.st", st):
        render_group_analytics(mas, reward_history)
