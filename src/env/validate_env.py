"""
validate_env.py

Utility for validating that an environment class or instance implements the required BaseEnvironment API and behaves correctly.
"""

import inspect
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.env.base_env import BaseEnvironment


def validate_environment(env_cls_or_instance, verbose=True):
    """
    Validate that the environment implements required methods/properties and basic functional checks.
    Accepts a class or instance.
    Returns True if valid, raises AssertionError (with details) if not.
    """
    # Accept class or instance
    if inspect.isclass(env_cls_or_instance):
        env = env_cls_or_instance()
    else:
        env = env_cls_or_instance
    # Check inheritance
    assert isinstance(
        env, BaseEnvironment
    ), f"{type(env)} does not inherit from BaseEnvironment"
    required_methods = ["reset", "step", "render", "get_state", "get_observation"]
    required_properties = ["action_space", "observation_space"]
    # Check methods
    for method in required_methods:
        assert hasattr(env, method), f"Missing required method: {method}"
        assert callable(getattr(env, method)), f"{method} is not callable"
    # Check properties
    for prop in required_properties:
        assert hasattr(env, prop), f"Missing required property: {prop}"
    # Functional checks
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Allow (obs, ...) return
    assert obs is not None, "reset() did not return an observation"
    action = env.action_space if not callable(env.action_space) else env.action_space()
    # Multi-agent support: if env has n_agents, pass a list of actions
    if hasattr(env, "n_agents") and isinstance(env.n_agents, int) and env.n_agents > 1:
        actions = [0 for _ in range(env.n_agents)]  # use 0 as default action
    else:
        actions = action if not isinstance(action, (list, tuple)) else action[0]
    # Try step
    try:
        step_result = env.step(actions)
    except Exception as e:
        raise AssertionError(f"step() failed: {e}")
    # Check get_state
    assert env.get_state() is not None, "get_state() returned None"
    # Check get_observation
    assert env.get_observation() is not None, "get_observation() returned None"
    if verbose:
        print(f"Environment {type(env).__name__} validated successfully.")
    return True
