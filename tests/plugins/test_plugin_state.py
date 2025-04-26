import os
import tempfile

from neuro_fuzzy_multiagent.core.plugins.plugin_state import PluginStateManager


def test_json_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PluginStateManager(tmpdir)
        state = {"counter": 5, "status": "ok"}
        mgr.save_json(state)
        loaded = mgr.load_json()
        assert loaded == state


def test_pickle_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PluginStateManager(tmpdir)
        obj = [1, 2, 3, {"x": 9}]
        mgr.save_pickle(obj)
        loaded = mgr.load_pickle()
        assert loaded == obj


def test_missing_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = PluginStateManager(tmpdir)
        assert mgr.load_json() is None
        assert mgr.load_pickle() is None
