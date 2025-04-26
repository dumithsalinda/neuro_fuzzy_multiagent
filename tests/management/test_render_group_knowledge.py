from dashboard.visualization import render_group_knowledge
from unittest.mock import patch, MagicMock
import streamlit as st


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
