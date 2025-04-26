"""
test_env_validation.py

Tests for the environment validation utility.
"""

import pytest
from src.env.validate_env import validate_environment
from src.env.environment_factory import EnvironmentFactory


@pytest.mark.parametrize(
    "env_name",
    [
        "multiagent_gridworld",
        "multiagent_gridworld_v2",
        "simple_discrete",
        "simple_continuous",
        "adversarial_gridworld",
        "multiagent_resource",
        "realworld_api",
        "noisy",
        "simple_abstract",
    ],
)
def test_registered_environment_valid(env_name):
    env = EnvironmentFactory.create(env_name)
    assert validate_environment(env, verbose=False)
