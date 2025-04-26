from src.agents.mock_agent import MockAgent


def test_mock_agent_act():
    agent = MockAgent()
    for obs in [0, 1, 42, -1]:
        action = agent.act(obs)
        assert action == 0
