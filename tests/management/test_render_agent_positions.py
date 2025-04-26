from dashboard.visualization import render_agent_positions
from unittest.mock import patch, MagicMock
import streamlit as st


def test_render_agent_positions_runs():
    agents = [MagicMock(group=0), MagicMock(group=1)]
    positions = [(0, 0), (1, 1)]
    with patch("dashboard.visualization.st", st):
        render_agent_positions(positions, agents)
