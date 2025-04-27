from neuro_fuzzy_multiagent.env.abstraction import NoisyEnvironment, SimpleEnvironment
from neuro_fuzzy_multiagent.env.adversarial_gridworld import AdversarialGridworldEnv
from neuro_fuzzy_multiagent.env.iot_sensor_fusion_env import IoTSensorFusionEnv

# Example registration (add others as needed)
from neuro_fuzzy_multiagent.env.multiagent_gridworld_env import MultiAgentGridworldEnv
from neuro_fuzzy_multiagent.env.multiagent_resource import MultiAgentResourceEnv
from neuro_fuzzy_multiagent.env.realworld_api_env import (
    RealWorldAPIEnv,  # If/when this file is moved, update accordingly.
)
from neuro_fuzzy_multiagent.env.simple_env import SimpleContinuousEnv, SimpleDiscreteEnv
from neuro_fuzzy_multiagent.env.multiagent_gridworld import (
    MultiAgentGridworldEnv as MultiAgentGridworldEnv2,
)


class EnvironmentFactory:
    """
    Factory for creating environments by name or config.
    Register new environments here for easy extension.
    """

    _registry = {}

    @classmethod
    def register(cls, name, env_cls):
        cls._registry[name] = env_cls

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown environment: {name}")
        return cls._registry[name](**kwargs)


EnvironmentFactory.register("multiagent_gridworld", MultiAgentGridworldEnv)
EnvironmentFactory.register(
    "multiagent_gridworld_v2", MultiAgentGridworldEnv2
)  # Disabled: MultiAgentGridworldEnv2 not defined
EnvironmentFactory.register("simple_discrete", SimpleDiscreteEnv)
EnvironmentFactory.register("simple_continuous", SimpleContinuousEnv)
EnvironmentFactory.register("adversarial_gridworld", AdversarialGridworldEnv)
EnvironmentFactory.register("multiagent_resource", MultiAgentResourceEnv)
EnvironmentFactory.register("realworld_api", RealWorldAPIEnv)
EnvironmentFactory.register("iot_sensor_fusion", IoTSensorFusionEnv)
EnvironmentFactory.register("noisy", NoisyEnvironment)
EnvironmentFactory.register("simple_abstract", SimpleEnvironment)
