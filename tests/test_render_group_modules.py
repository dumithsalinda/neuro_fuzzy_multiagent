from dashboard.visualization import render_group_modules
from unittest.mock import patch
import streamlit as st

def test_render_group_modules_with_data():
    group_modules = {0: {"param": 1}, 1: {"param": 2}}
    with patch("dashboard.visualization.st", st):
        render_group_modules(group_modules)

def test_render_group_modules_no_data():
    with patch("dashboard.visualization.st", st):
        render_group_modules({})
