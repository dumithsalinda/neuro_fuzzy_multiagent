import pytest
from src.core.management.agent_manager import AgentManager
from src.core.management.message_bus import MessageBus
from src.core.agents.agent import Agent

from .dummy_agent import DummyAgent

def minimal_agent_config(name):
    return {
        'agent_type': 'DummyAgent',
        'nn_config': {'input_dim': 1, 'hidden_dim': 1, 'output_dim': 1},
        'fis_config': None,
        'meta_controller': {},
        'universal_fuzzy_layer': None,
        'name': name
    }

def test_dynamic_agent_join_leave():
    bus = MessageBus()
    manager = AgentManager(bus=bus)
    config_a = minimal_agent_config('A')
    config_b = minimal_agent_config('B')
    agent_a = manager.add_agent(config_a, group='G1')
    agent_b = manager.add_agent(config_b, group='G1')
    # Both agents should be in group
    group_agents = manager.get_agents(group='G1')
    assert agent_a in group_agents and agent_b in group_agents
    # Remove one agent
    manager.remove_agent(agent_a)
    group_agents = manager.get_agents(group='G1')
    assert agent_a not in group_agents and agent_b in group_agents
    # Messaging still works for remaining agent
    agent_b.send_message({'msg': 'hi'}, recipient=agent_b)
    msg, sender = agent_b.last_message
    assert msg['msg'] == 'hi'
