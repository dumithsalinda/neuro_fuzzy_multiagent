from dashboard.layout import render_layout
from unittest.mock import patch
import streamlit as st


def test_render_layout_runs():
    with patch("dashboard.layout.st", st):
        render_layout()
