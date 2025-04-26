import numpy as np
import pytest
from neuro_fuzzy_multiagent.env.environment_factory import EnvironmentFactory


def test_simple_discrete_extract_features():
    env = EnvironmentFactory.create("simple_discrete", n_states=5, n_actions=2)
    obs = env.reset()
    features = env.extract_features()
    assert np.allclose(features, obs)
    assert isinstance(features, np.ndarray)


def test_simple_continuous_extract_features():
    env = EnvironmentFactory.create("simple_continuous")
    obs = env.reset()
    features = env.extract_features()
    assert np.allclose(features, obs)
    assert isinstance(features, np.ndarray)


def test_multiagent_gridworld_extract_features():
    env = EnvironmentFactory.create(
        "multiagent_gridworld", grid_size=4, n_agents=2, n_obstacles=1
    )
    obs = env.reset()
    features = env.extract_features()
    assert np.allclose(features, obs)
    assert isinstance(features, np.ndarray)


def test_adversarial_gridworld_extract_features():
    env = EnvironmentFactory.create(
        "adversarial_gridworld", grid_size=4, n_pursuers=1, n_evaders=1, n_obstacles=0
    )
    obs = env.reset()
    features = env.extract_features()
    # Adversarial env returns list of arrays for obs/features
    assert isinstance(features, list)
    assert all(isinstance(f, np.ndarray) for f in features)


def test_multiagent_resource_extract_features():
    env = EnvironmentFactory.create(
        "multiagent_resource", grid_size=4, n_agents=2, n_resources=2
    )
    obs = env.reset()
    features = env.extract_features()
    # Resource env returns list of arrays for obs/features
    assert isinstance(features, list)
    assert all(isinstance(f, np.ndarray) for f in features)


def test_perceive_matches_observation():
    env = EnvironmentFactory.create("simple_continuous")
    obs = env.get_observation()
    perceived = env.perceive()
    assert np.allclose(perceived, obs)
