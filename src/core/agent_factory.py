import json
import yaml
from src.core.agent import NeuroFuzzyAgent
from src.core.meta_controller import MetaController
from src.core.universal_fuzzy_layer import UniversalFuzzyLayer

def create_agent_from_config(config):
    """
    Create an agent from a config dict (JSON/YAML-parsed).
    Supports NeuroFuzzyAgent and dynamic attachment of fuzzy/meta layers.
    """
    agent_type = config.get('agent_type', 'NeuroFuzzyAgent')
    nn_config = config.get('nn_config', {})
    fis_config = config.get('fis_config', None)
    meta_cfg = config.get('meta_controller', None)
    universal_fuzzy_cfg = config.get('universal_fuzzy_layer', None)

    meta_controller = MetaController(**meta_cfg) if meta_cfg else None
    universal_fuzzy_layer = None
    if universal_fuzzy_cfg:
        universal_fuzzy_layer = UniversalFuzzyLayer(**universal_fuzzy_cfg)

    if agent_type == 'NeuroFuzzyAgent':
        agent = NeuroFuzzyAgent(
            nn_config=nn_config,
            fis_config=fis_config,
            universal_fuzzy_layer=universal_fuzzy_layer,
            meta_controller=meta_controller
        )
    elif agent_type == 'DummyAgent':
        # Import DummyAgent from the test module if available, else fallback to Agent
        from tests.dummy_agent import DummyAgent
        agent = DummyAgent(model=None)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    return agent

def load_agent_config(path):
    """
    Load agent config from a YAML or JSON file.
    """
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")
    return config


def create_agent_from_file(path):
    config = load_agent_config(path)
    return create_agent_from_config(config)
