"""
Auto-generate PLUGIN_DOCS.md by scanning all plugin registries and extracting docstrings and config signatures.
Run: python generate_plugin_docs.py
"""
import os
import inspect
import importlib
from pathlib import Path

# Registry locations
REGISTRIES = [
    ("Environments", "src.env.registry", "get_registered_environments"),
    ("Agents", "src.core.agent_registry", "get_registered_agents"),
    ("Neural Networks", "src.core.neural_network", "get_registered_networks"),
    ("Sensors", "src.plugins.registry", "get_registered_sensors"),
    ("Actuators", "src.plugins.registry", "get_registered_actuators"),
]

out_lines = ["# Plugin API Reference\n"]

for section, mod_path, func_name in REGISTRIES:
    try:
        mod = importlib.import_module(mod_path)
        registry = getattr(mod, func_name)()
    except Exception as e:
        out_lines.append(f"\n## {section}\nCould not load registry: {e}\n")
        continue
    out_lines.append(f"\n## {section}\n")
    if not registry:
        out_lines.append("*(No plugins found)*\n")
        continue
    for name, cls in registry.items():
        out_lines.append(f"### {name}\n")
        doc = inspect.getdoc(cls) or "No docstring."
        out_lines.append(f"{doc}\n")
        # List __init__ signature for config
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            if params:
                out_lines.append("**Config options:**\n")
                for p in params:
                    out_lines.append(f"- `{p.name}`: {p.annotation if p.annotation != inspect._empty else 'Any'} (default: {p.default if p.default != inspect._empty else 'required'})")
            out_lines.append("")
        except Exception:
            pass

with open("PLUGIN_DOCS.md", "w") as f:
    f.write("\n".join(out_lines))

print("PLUGIN_DOCS.md generated.")
