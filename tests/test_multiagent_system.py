import numpy as np
import pytest
from src.core.multiagent_system import MultiAgentSystem
from src.core.agent import Agent

class DummyAgent(Agent):
    def __init__(self):
        super().__init__(model=None)
        self.received = []
    def act(self, observation, state=None):
        return observation
    def reset(self):
        self.received = []

def test_multiagent_step_and_reset():
    agents = [DummyAgent() for _ in range(3)]
    mas = MultiAgentSystem(agents)
    obs = [np.ones(2) * i for i in range(3)]
    actions = mas.step(obs)
    assert len(actions) == 3
    for i, action in enumerate(actions):
        assert np.all(action == obs[i])
    mas.reset()
    for agent in agents:
        assert agent.received == []

def test_multiagent_direct_message():
    agents = [DummyAgent() for _ in range(2)]
    mas = MultiAgentSystem(agents)
    mas.send_message(0, 1, "direct-msg", msg_type="INFO")
    msgs = mas.get_messages(1)
    assert len(msgs) == 1
    assert msgs[0]["from"] == 0
    assert msgs[0]["type"] == "INFO"
    assert msgs[0]["content"] == "direct-msg"
    assert mas.get_messages(0) == []

def test_multiagent_broadcast_message():
    agents = [DummyAgent() for _ in range(3)]
    mas = MultiAgentSystem(agents)
    mas.broadcast_message(0, "hello-all", msg_type="BROADCAST")
    for i in range(1, 3):
        msgs = mas.get_messages(i)
        assert len(msgs) == 1
        assert msgs[0]["from"] == 0
        assert msgs[0]["type"] == "BROADCAST"
        assert msgs[0]["content"] == "hello-all"
    assert mas.get_messages(0) == []

def test_multiagent_message_type_filtering():
    agents = [DummyAgent() for _ in range(2)]
    mas = MultiAgentSystem(agents)
    mas.send_message(0, 1, "foo", msg_type="INFO")
    mas.send_message(0, 1, "bar", msg_type="DATA")
    msgs_info = mas.get_messages(1, msg_type="INFO")
    msgs_data = mas.get_messages(1, msg_type="DATA")
    assert len(msgs_info) == 1 and msgs_info[0]["content"] == "foo"
    assert len(msgs_data) == 1 and msgs_data[0]["content"] == "bar"

def test_multiagent_message_history():
    agents = [DummyAgent() for _ in range(2)]
    mas = MultiAgentSystem(agents)
    mas.send_message(0, 1, "foo", msg_type="INFO")
    mas.broadcast_message(0, "bar", msg_type="BROADCAST")
    history = mas.get_message_history(1)
    assert any(msg["type"] == "INFO" for msg in history)
    assert any(msg["type"] == "BROADCAST" for msg in history)
