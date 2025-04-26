from neuro_fuzzy_multiagent.dashboard.visualization import render_som_grid_with_agents
from unittest.mock import patch, MagicMock
import streamlit as st
import numpy as np


def test_render_som_grid_with_agents_handles_no_weights():
    mas = MagicMock()
    som = MagicMock(get_weights=lambda: None)
    with patch("dashboard.visualization.st", st):
        render_som_grid_with_agents(mas, som)


def test_render_som_grid_with_agents_runs():
    class DummySOM:
        def get_weights(self):
            return np.ones((2, 2, 3))

        som_shape = (2, 2)

        def predict(self, obs):
            return [(0, 0), (1, 1)]

    mas = MagicMock()
    mas.agents = [
        MagicMock(last_observation=np.zeros(3), group=0),
        MagicMock(last_observation=np.zeros(3), group=1),
    ]
    som = DummySOM()
    with patch("dashboard.visualization.st", st):
        render_som_grid_with_agents(mas, som)
