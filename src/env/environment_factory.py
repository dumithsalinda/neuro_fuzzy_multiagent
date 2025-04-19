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

# Example registration (add others as needed)
from src.env.multiagent_gridworld_env import MultiAgentGridworldEnv
from src.env.multiagent_gridworld import MultiAgentGridworldEnv as MultiAgentGridworldEnv2
from src.env.simple_env import SimpleDiscreteEnv, SimpleContinuousEnv
from src.env.adversarial_gridworld import AdversarialGridworldEnv
from src.env.multiagent_resource import MultiAgentResourceEnv
from src.env.realworld_api_env import RealWorldAPIEnv
from src.env.iot_sensor_fusion_env import IoTSensorFusionEnv
from src.environment.abstraction import NoisyEnvironment, SimpleEnvironment

EnvironmentFactory.register("multiagent_gridworld", MultiAgentGridworldEnv)
EnvironmentFactory.register("multiagent_gridworld_v2", MultiAgentGridworldEnv2)
EnvironmentFactory.register("simple_discrete", SimpleDiscreteEnv)
EnvironmentFactory.register("simple_continuous", SimpleContinuousEnv)
EnvironmentFactory.register("adversarial_gridworld", AdversarialGridworldEnv)
EnvironmentFactory.register("multiagent_resource", MultiAgentResourceEnv)
EnvironmentFactory.register("realworld_api", RealWorldAPIEnv)
EnvironmentFactory.register("iot_sensor_fusion", IoTSensorFusionEnv)
EnvironmentFactory.register("noisy", NoisyEnvironment)
EnvironmentFactory.register("simple_abstract", SimpleEnvironment)
