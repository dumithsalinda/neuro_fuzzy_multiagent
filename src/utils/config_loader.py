"""
Utility for loading agent neural network config from YAML or JSON.
Supports plug-and-play neural network selection for NeuroFuzzyHybrid.
"""

import yaml
import json


def load_nn_config(config_path):
    """
    Load nn_config from YAML or JSON file.
    Returns the nn_config dict (ready for NeuroFuzzyHybrid).
    Activation functions must be specified as strings (e.g., 'tanh', 'relu').
    """
    if config_path.endswith((".yaml", ".yml")):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file type: {config_path}")
    if "nn_config" not in config:
        raise KeyError("Config file must contain 'nn_config' at the top level.")
    # Validate activation is a string if present
    act = config["nn_config"].get("activation", None)
    if act is not None and not isinstance(act, str):
        raise ValueError(
            "'activation' must be a string such as 'tanh', 'relu', or 'sigmoid'."
        )
    return config["nn_config"]


# Example usage:
# from utils.config_loader import load_nn_config
# nn_config = load_nn_config('config/nn_config_example.yaml')
# agent = NeuroFuzzyHybrid(nn_config, fis_config=...)
