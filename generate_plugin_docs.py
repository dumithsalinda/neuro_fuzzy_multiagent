"""
Auto-generate PLUGIN_DOCS.md by scanning all plugin registries and extracting docstrings and config signatures.
Run: python generate_plugin_docs.py
"""

import os
import inspect
import importlib
from pathlib import Path

import os
import inspect
from pathlib import Path
import sys

# Add src to sys.path for import compatibility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import os
import inspect
from pathlib import Path
import sys
import importlib
import glob

# Add src to sys.path for import compatibility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))


# Dynamically import all .py files in these directories (except __init__.py)
def import_all_modules_from_dir(package, directory):
    abs_dir = os.path.abspath(directory)
    for file in glob.glob(os.path.join(abs_dir, "*.py")):
        base = os.path.basename(file)
        if base.startswith("__") or base == "__init__.py":
            continue
        module_name = os.path.splitext(base)[0]
        importlib.import_module(f"{package}.{module_name}")


import_all_modules_from_dir("core.agents", "src/core/agents")
import_all_modules_from_dir("core.environments", "src/core/environments")
import_all_modules_from_dir("core.sensors", "src/core/sensors")
import_all_modules_from_dir("core.actuators", "src/core/actuators")
import_all_modules_from_dir("core.plugins", "src/core/plugins")

from core.plugins.registration_utils import PLUGIN_REGISTRIES

out_lines = ["# Plugin API Documentation\n"]

for plugin_type, registry in PLUGIN_REGISTRIES.items():
    section = plugin_type.capitalize() + " Plugins"
    out_lines.append(f"\n## {section}\n")
    if not registry:
        out_lines.append("_No plugins registered._\n")
        continue
    for name, cls in registry.items():
        out_lines.append(f"### {name}\n")
        version = getattr(cls, "__version__", None)
        if version:
            out_lines.append(f"**Version:** `{version}`  ")
        doc = inspect.getdoc(cls) or "No docstring."
        out_lines.append(f"{doc}\n")
        # List __init__ signature for config
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            if params:
                out_lines.append("**Config options:**\n")
                for p in params:
                    ann = p.annotation if p.annotation != inspect._empty else "Any"
                    default = p.default if p.default != inspect._empty else "required"
                    out_lines.append(f"- `{p.name}`: {ann} (default: {default})")
            out_lines.append("")
        except Exception:
            pass

with open("PLUGIN_DOCS.md", "w") as f:
    f.write("\n".join(out_lines))

print("PLUGIN_DOCS.md generated.")
