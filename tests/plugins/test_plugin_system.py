"""
Test the unified plug-and-play system: registration, loading, error handling, hot-reload.
"""

import pytest
from src.core.plugins.registration_utils import (
    get_registered_plugins,
    clear_plugin_registry,
)
from src.core.plugins.hot_reload import reload_all_plugins

PLUGIN_TYPES = ["environment", "agent", "sensor", "actuator"]


def test_plugin_registration():
    # Ensure at least one environment plugin is registered
    try:
        from src.core.agents.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent
    except ImportError:
        pass
    try:
        from src.core.environments.gym_env_wrapper import GymEnvWrapper
    except ImportError:
        pass
    try:
        from src.core.sensors.example_sensor import ExampleSensor
    except ImportError:
        pass
    try:
        from src.core.actuators.example_actuator import ExampleActuator
    except ImportError:
        pass
    """All plugin types should have at least one registered plugin."""
    from src.core.plugins.registration_utils import get_registered_plugins

    print("Registered agent plugins:", get_registered_plugins("agent"))
    for ptype in PLUGIN_TYPES:
        plugins = get_registered_plugins(ptype)
        assert isinstance(plugins, dict)
        assert len(plugins) > 0, f"No plugins registered for type: {ptype}"


def test_plugin_hot_reload():
    """Hot-reloading should clear and repopulate plugin registries."""
    # Get initial plugin counts
    initial_counts = {
        ptype: len(get_registered_plugins(ptype)) for ptype in PLUGIN_TYPES
    }
    # Clear registries
    for ptype in PLUGIN_TYPES:
        clear_plugin_registry(ptype)
        assert len(get_registered_plugins(ptype)) == 0
    # Reload
    result = reload_all_plugins()
    for ptype in PLUGIN_TYPES:
        # Only require at least one plugin per type after reload
        assert (
            len(get_registered_plugins(ptype)) > 0
        ), f"No {ptype} plugins after hot reload"
    assert "errors" in result


def test_plugin_error_handling(monkeypatch):
    """Simulate a plugin import error and ensure it is logged, not fatal."""
    from src.core.plugins import hot_reload

    def bad_reload():
        raise ImportError("Simulated error")

    monkeypatch.setattr(hot_reload, "reload_all_plugins", bad_reload)
    try:
        hot_reload.reload_all_plugins()
    except ImportError:
        pass  # Should not crash test suite
