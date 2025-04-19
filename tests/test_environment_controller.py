import unittest
from src.env.environment_controller import EnvironmentController
from src.env.environment_factory import EnvironmentFactory
from src.env.iot_sensor_fusion_env import IoTSensorFusionEnv
from src.env.multiagent_gridworld_env import MultiAgentGridworldEnv
from src.core.agent import Agent
from src.core.multiagent_system import MultiAgentSystem

class DummyAgent(Agent):
    def __init__(self):
        super().__init__(model=None)
        self.reset_called = False
    def reset(self):
        self.reset_called = True
        super().reset()

class TestEnvironmentController(unittest.TestCase):
    def setUp(self):
        self.agents = [DummyAgent(), DummyAgent()]
        self.mas = MultiAgentSystem(self.agents)
        self.controller = EnvironmentController('multiagent_gridworld', agents=self.agents, multiagent_system=self.mas, grid_size=5, n_agents=2)

    def test_switch_environment_and_state_transfer(self):
        # Initial env
        env1 = self.controller.get_env()
        self.assertIsInstance(env1, MultiAgentGridworldEnv)
        # Switch to IoT env
        env2 = self.controller.switch_environment('iot_sensor_fusion', grid_size=6, n_agents=2)
        self.assertIsInstance(env2, IoTSensorFusionEnv)
        # State transfer (default: agent.reset called)
        for agent in self.agents:
            self.assertTrue(agent.reset_called)

    def test_custom_transfer_fn(self):
        called = []
        def custom_transfer(agent, old_env, new_env):
            called.append((agent, type(old_env), type(new_env)))
        self.controller.switch_environment('iot_sensor_fusion', transfer_fn=custom_transfer, grid_size=6, n_agents=2)
        self.assertEqual(len(called), 2)
        for entry in called:
            self.assertEqual(entry[1], MultiAgentGridworldEnv)
            self.assertEqual(entry[2], IoTSensorFusionEnv)

if __name__ == '__main__':
    unittest.main()
