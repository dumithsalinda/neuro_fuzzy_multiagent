import pytest
from neuro_fuzzy_multiagent.env.environment_factory import EnvironmentFactory


def test_factory_creates_noisy_env():
    env = EnvironmentFactory.create("noisy", dim=4, noise_std=0.1)
    obs = env.reset()
    assert obs.shape == (4,)
    assert hasattr(env, "step")
    assert hasattr(env, "extract_features")


def test_factory_creates_simple_abstract_env():
    env = EnvironmentFactory.create("simple_abstract", dim=2)
    obs = env.reset()
    assert obs.shape == (2,)
    assert hasattr(env, "step")
    assert hasattr(env, "extract_features")


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        EnvironmentFactory.create("does_not_exist")
