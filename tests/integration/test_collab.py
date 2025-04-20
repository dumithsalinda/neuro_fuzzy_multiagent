import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from dashboard.collab import collaborative_experiments_panel

def test_collaborative_experiments_panel_runs():
    st.session_state.clear()
    with patch("dashboard.collab.st", st), \
         patch("dashboard.collab.st.sidebar.expander"), \
         patch("dashboard.collab.st.markdown"), \
         patch("dashboard.collab.st.file_uploader", return_value=None), \
         patch("dashboard.collab.st.text_input", return_value=""), \
         patch("dashboard.collab.st.button", return_value=False):
        collaborative_experiments_panel()
