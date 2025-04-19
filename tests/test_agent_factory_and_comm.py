import os
import tempfile
import yaml
import json
from src.core.agent_factory import create_agent_from_config, create_agent_from_file
from src.core.agent import Agent

from tests.dummy_agent import DummyAgent

def test_create_agent_from_yaml():
    config = {
        'agent_type': 'NeuroFuzzyAgent',
        'nn_config': {'input_dim': 2, 'hidden_dim': 2, 'output_dim': 1},
        'fis_config': None,
        'meta_controller': {},
        'universal_fuzzy_layer': None
    }
    with tempfile.NamedTemporaryFile('w+', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        fname = f.name
    agent = create_agent_from_file(fname)
    assert hasattr(agent, 'set_mode')
    os.remove(fname)

def test_create_agent_from_json():
    config = {
        'agent_type': 'NeuroFuzzyAgent',
        'nn_config': {'input_dim': 2, 'hidden_dim': 2, 'output_dim': 1},
        'fis_config': None,
        'meta_controller': {},
        'universal_fuzzy_layer': None
    }
    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as f:
        json.dump(config, f)
        fname = f.name
    agent = create_agent_from_file(fname)
    assert hasattr(agent, 'set_mode')
    os.remove(fname)

def test_agent_communication():
    a1 = Agent(model=None)
    a2 = Agent(model=None)
    a1.send_message({'msg': 'hello', 'from': 'a1'}, recipient=a2)
    msg, sender = a2.last_message
    assert msg == {'msg': 'hello', 'from': 'a1'}
    assert isinstance(sender, Agent)
    msg2, sender2 = a2.message_inbox[-1]
    assert msg2 == {'msg': 'hello', 'from': 'a1'}
    # Test direct call with no bus
    a2.send_message({'msg': 'pong', 'from': 'a2'}, recipient=a1)
    msg3, sender3 = a1.last_message
    assert msg3['from'] == 'a2'
    assert isinstance(sender3, Agent)
