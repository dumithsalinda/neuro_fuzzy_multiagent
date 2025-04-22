"""
CLI tool to validate all registered plugins for dependency and interface compliance.
"""
import sys
import os
import importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core.plugins.registration_utils import get_registered_plugins

PLUGIN_TYPES = ["agent", "environment", "sensor", "actuator"]
REQUIRED_AGENT_METHODS = ["act", "explain_action"]
REQUIRED_ENV_METHODS = ["reset", "step"]
REQUIRED_SENSOR_METHODS = ["read"]
REQUIRED_ACTUATOR_METHODS = ["actuate"]

INTERFACE_MAP = {
    "agent": REQUIRED_AGENT_METHODS,
    "environment": REQUIRED_ENV_METHODS,
    "sensor": REQUIRED_SENSOR_METHODS,
    "actuator": REQUIRED_ACTUATOR_METHODS,
}

def validate_plugin_interface(plugin_type, cls):
    """Check that the plugin class implements required methods."""
    missing = []
    for method in INTERFACE_MAP.get(plugin_type, []):
        if not hasattr(cls, method):
            missing.append(method)
    return missing

def main():
    failed = False
    print("Validating plugin interfaces...")
    for ptype in PLUGIN_TYPES:
        plugins = get_registered_plugins(ptype)
        for name, cls in plugins.items():
            missing = validate_plugin_interface(ptype, cls)
            if missing:
                print(f"[FAIL] {ptype} plugin '{name}' is missing methods: {missing}")
                failed = True
            else:
                print(f"[OK] {ptype} plugin '{name}' implements all required methods.")
    if failed:
        print("\nSome plugins failed validation.")
        sys.exit(1)
    else:
        print("\nAll plugins passed validation.")
        sys.exit(0)

if __name__ == "__main__":
    main()
