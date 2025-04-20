import json
import yaml
from src.core.agents.agent import NeuroFuzzyAgent
from src.core.management.meta_controller import MetaController
from src.core.neural_networks.universal_fuzzy_layer import UniversalFuzzyLayer
from src.core.agents.neuro_fuzzy_fusion_agent import NeuroFuzzyFusionAgent
from src.core.agents.dqn_agent import DQNAgent
from src.core.agents.multimodal_dqn_agent import MultiModalDQNAgent
from src.core.agents.anfis_agent import NeuroFuzzyANFISAgent
from src.core.agents.multimodal_fusion_agent import MultiModalFusionAgent


def create_agent_from_config(config):
    """
    Create an agent from a config dict (JSON/YAML-parsed).
    Supports NeuroFuzzyAgent and dynamic attachment of fuzzy/meta layers.
    """
    agent_type = config.get("agent_type", "NeuroFuzzyAgent")
    nn_config = config.get("nn_config", {})
    fis_config = config.get("fis_config", None)
    meta_cfg = config.get("meta_controller", None)
    universal_fuzzy_cfg = config.get("universal_fuzzy_layer", None)

    meta_controller = MetaController(**meta_cfg) if meta_cfg else None
    universal_fuzzy_layer = None
    if universal_fuzzy_cfg:
        universal_fuzzy_layer = UniversalFuzzyLayer(**universal_fuzzy_cfg)

    if agent_type == "NeuroFuzzyAgent":
        agent = NeuroFuzzyAgent(
            nn_config=nn_config,
            fis_config=fis_config,
            universal_fuzzy_layer=universal_fuzzy_layer,
            meta_controller=meta_controller,
        )
    elif agent_type == "NeuroFuzzyFusionAgent":
        # Required: input_dims, hidden_dim, output_dim
        required = ["input_dims", "hidden_dim", "output_dim"]
        for r in required:
            if r not in config:
                raise ValueError(
                    f"Missing required config field '{r}' for NeuroFuzzyFusionAgent"
                )
        agent = NeuroFuzzyFusionAgent(
            input_dims=config["input_dims"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            fusion_type=config.get("fusion_type", "concat"),
            fuzzy_config=config.get("fuzzy_config", None),
            fusion_alpha=config.get("fusion_alpha", 0.5),
            device=config.get("device", None),
        )
    elif agent_type == "DQNAgent":
        required = ["state_dim", "action_dim"]
        for r in required:
            if r not in config:
                raise ValueError(f"Missing required config field '{r}' for DQNAgent")
        agent = DQNAgent(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            alpha=config.get("alpha", 1e-3),
            gamma=config.get("gamma", 0.99),
            epsilon=config.get("epsilon", 0.1),
        )
    elif agent_type == "MultiModalDQNAgent":
        required = ["input_dims", "action_dim"]
        for r in required:
            if r not in config:
                raise ValueError(
                    f"Missing required config field '{r}' for MultiModalDQNAgent"
                )
        agent = MultiModalDQNAgent(
            input_dims=config["input_dims"],
            action_dim=config["action_dim"],
            alpha=config.get("alpha", 1e-3),
            gamma=config.get("gamma", 0.99),
            epsilon=config.get("epsilon", 0.1),
        )
    elif agent_type == "NeuroFuzzyANFISAgent":
        required = ["input_dim", "n_rules"]
        for r in required:
            if r not in config:
                raise ValueError(
                    f"Missing required config field '{r}' for NeuroFuzzyANFISAgent"
                )
        agent = NeuroFuzzyANFISAgent(
            input_dim=config["input_dim"],
            n_rules=config["n_rules"],
            lr=config.get("lr", 0.01),
            buffer_size=config.get("buffer_size", 100),
            replay_enabled=config.get("replay_enabled", True),
            replay_batch=config.get("replay_batch", 8),
            meta_update_fn=config.get("meta_update_fn", None),
        )
    elif agent_type == "MultiModalFusionAgent":
        required = ["input_dims", "hidden_dim", "output_dim"]
        for r in required:
            if r not in config:
                raise ValueError(
                    f"Missing required config field '{r}' for MultiModalFusionAgent"
                )
        agent = MultiModalFusionAgent(
            input_dims=config["input_dims"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            fusion_type=config.get("fusion_type", "concat"),
            lr=config.get("lr", 1e-3),
            gamma=config.get("gamma", 0.99),
        )
    elif agent_type == "DummyAgent":
        from agents.dummy_agent import DummyAgent

        agent = DummyAgent(model=None)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    return agent


def load_agent_config(path):
    """
    Load agent config from a YAML or JSON file.
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config file must be .yaml, .yml, or .json")
    return config


def create_agent_from_file(path):
    config = load_agent_config(path)
    return create_agent_from_config(config)
