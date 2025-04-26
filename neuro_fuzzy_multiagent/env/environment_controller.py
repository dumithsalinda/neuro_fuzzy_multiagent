# from neuro_fuzzy_multiagent.env.mock_environment import MockEnvironmentFactory
from neuro_fuzzy_multiagent.env.environment_factory import EnvironmentFactory


class EnvironmentController:
    """
    Manages the current environment instance and allows hot-swapping between environments at runtime.
    Supports agent state transfer during environment switch.
    Optionally integrates with a MultiAgentSystem.
    """

    def __init__(self, env_name, agents=None, multiagent_system=None, **env_kwargs):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env = EnvironmentFactory.create(env_name, **env_kwargs)
        self.agents = agents
        self.multiagent_system = multiagent_system

    def get_env(self):
        return self.env

    def switch_environment(
        self, new_env_name, transfer_state=True, transfer_fn=None, **new_env_kwargs
    ):
        """
        Switch environment at runtime. Optionally transfer agent and environment state.
        transfer_fn: custom function(agent, old_env, new_env) for state transfer.
        """
        old_env = self.env
        self.env_name = new_env_name
        self.env_kwargs = new_env_kwargs
        self.env = EnvironmentFactory.create(new_env_name, **new_env_kwargs)
        if transfer_state and self.agents is not None:
            self._transfer_agent_state(old_env, self.env, transfer_fn=transfer_fn)
        return self.env

    def reset_environment(self, **reset_kwargs):
        return self.env.reset(**reset_kwargs)

    def _transfer_agent_state(self, old_env, new_env, transfer_fn=None):
        """
        Transfer agent state from old_env to new_env.
        By default, resets agent and optionally copies last observation/action.
        Custom strategies can be implemented via transfer_fn.
        """
        if self.multiagent_system is not None:
            agents = self.multiagent_system.agents
        elif self.agents is not None:
            agents = self.agents
        else:
            return
        for agent in agents:
            if transfer_fn is not None:
                transfer_fn(agent, old_env, new_env)
            else:
                # Default: reset agent, optionally copy last observation/action if compatible
                agent.reset()
                # Optionally: agent.last_observation = old_env.get_observation() or similar


# Example integration with MultiAgentSystem:
# from neuro_fuzzy_multiagent.core.multiagent_system import MultiAgentSystem
# agents = [Agent(...), ...]
# mas = MultiAgentSystem(agents)
# controller = EnvironmentController('multiagent_gridworld', agents=agents, multiagent_system=mas, grid_size=5, n_agents=2)
# controller.switch_environment('iot_sensor_fusion', grid_size=6, n_agents=3)
# # Agent state transfer can be customized with transfer_fn
