"""
Entry point for the Multi-Agent System Dashboard.
All dashboard logic is delegated to modules in the dashboard/ folder.
"""

import streamlit as st
from dashboard.main import main

if __name__ == "__main__":
    main()
