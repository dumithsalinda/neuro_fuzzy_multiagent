"""
Basic tests for dashboard modules: simulation, visualization, tables, analytics.
Uses pytest and Streamlit's testing utilities where possible.
"""
import pytest
import types
import pandas as pd
from dashboard import simulation, visualization, tables, analytics

def test_som_group_agents_handles_missing_state(monkeypatch):
    # Should warn if agents or obs missing
    class DummySt:
        session_state = {}
        @staticmethod
        def warning(msg):
            assert "No agents" in msg
    monkeypatch.setattr(simulation, "st", DummySt)
    simulation.som_group_agents()

def test_render_knowledge_table_handles_missing_attrs(monkeypatch):
    # Should warn if agent missing expected attributes
    class DummySt:
        def header(self, msg): pass
        def table(self, data): assert isinstance(data, list)
        def warning(self, msg): assert "missing expected attributes" in msg
        def info(self, msg): pass
    monkeypatch.setattr(tables, "st", DummySt())
    tables.render_knowledge_table([object()])

def test_render_batch_analytics_empty(monkeypatch):
    # Should warn if df empty
    class DummySt:
        def warning(self, msg): assert "No batch experiment data" in msg
        def subheader(self, msg): pass
        def dataframe(self, df): pass
        def markdown(self, txt): pass
        def write(self, x): pass
        def line_chart(self, x): pass
        def pyplot(self, fig): pass
    monkeypatch.setattr(analytics, "st", DummySt())
    analytics.render_batch_analytics(pd.DataFrame())

def test_render_group_decisions_log_handles_malformed(monkeypatch):
    # Should warn if malformed group decision
    class DummySt:
        def header(self, msg): pass
        def markdown(self, msg, unsafe_allow_html=False): pass
        def warning(self, msg): assert "Malformed group decision" in msg or "missing expected attributes" in msg
        def info(self, msg): pass
    monkeypatch.setattr(tables, "st", DummySt())
    tables.render_group_decisions_log([object()])

def test_render_advanced_metrics_empty(monkeypatch):
    class DummySt:
        def warning(self, msg): assert "No data for advanced metrics" in msg
        def subheader(self, msg): pass
        def write(self, x): pass
        def info(self, x): pass
        def pyplot(self, fig): pass
    monkeypatch.setattr(analytics, "st", DummySt())
    analytics.render_advanced_metrics(pd.DataFrame())
