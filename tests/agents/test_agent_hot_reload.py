import os
import tempfile
import yaml
from src.core.management.agent_manager import AgentManager
from src.core.management.message_bus import MessageBus


def test_agent_hot_reload():
    bus = MessageBus()
    manager = AgentManager(bus=bus)
    config = {
        "agent_type": "NeuroFuzzyAgent",
        "nn_config": {"input_dim": 1, "hidden_dim": 2, "output_dim": 1},
        "fis_config": None,
        "meta_controller": {},
        "universal_fuzzy_layer": None,
    }
    agent = manager.add_agent(config)
    # Save a new config with different NN params
    new_config = {
        "agent_type": "NeuroFuzzyAgent",
        "nn_config": {"input_dim": 3, "hidden_dim": 4, "output_dim": 2},
        "fis_config": None,
        "meta_controller": {},
        "universal_fuzzy_layer": None,
    }
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as f:
        yaml.dump(new_config, f)
        fname = f.name
    # Reload config at runtime
    manager.reload_agent_config(agent, fname)
    # Check that the neural network config was updated
    assert agent.model.nn.input_dim == 3
    assert agent.model.nn.hidden_dim == 4
    assert agent.model.nn.output_dim == 2
    os.remove(fname)
