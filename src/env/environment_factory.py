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
from .multiagent_gridworld_env import MultiAgentGridworldEnv
from .multiagent_gridworld import MultiAgentGridworldEnv as MultiAgentGridworldEnv2
from .simple_env import SimpleDiscreteEnv, SimpleContinuousEnv
from .adversarial_gridworld import AdversarialGridworldEnv
from .multiagent_resource import MultiAgentResourceEnv

EnvironmentFactory.register("multiagent_gridworld", MultiAgentGridworldEnv)
EnvironmentFactory.register("multiagent_gridworld_v2", MultiAgentGridworldEnv2)
EnvironmentFactory.register("simple_discrete", SimpleDiscreteEnv)
EnvironmentFactory.register("simple_continuous", SimpleContinuousEnv)
EnvironmentFactory.register("adversarial_gridworld", AdversarialGridworldEnv)
EnvironmentFactory.register("multiagent_resource", MultiAgentResourceEnv)
