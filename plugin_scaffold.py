import argparse
import os
from datetime import datetime

TEMPLATES = {
    "environment": '''"""
Environment Plugin Template
Created: {date}
"""
from neuro_fuzzy_multiagent.env.base_environment import BaseEnvironment

class {name}(BaseEnvironment):
    """Custom environment plugin."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        pass

    def reset(self):
        pass
''',
    "agent": '''"""
Agent Plugin Template
Created: {date}
"""
from neuro_fuzzy_multiagent.agents.base_agent import BaseAgent

class {name}(BaseAgent):
    """Custom agent plugin."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, observation):
        pass
''',
    "sensor": '''"""
Sensor Plugin Template
Created: {date}
"""
from neuro_fuzzy_multiagent.plugins.base_sensor import BaseSensor

class {name}(BaseSensor):
    """Custom sensor plugin."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sense(self, *args, **kwargs):
        pass
''',
    "actuator": '''"""
Actuator Plugin Template
Created: {date}
"""
from neuro_fuzzy_multiagent.plugins.base_actuator import BaseActuator

class {name}(BaseActuator):
    """Custom actuator plugin."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def actuate(self, *args, **kwargs):
        pass
''',
}

PLUGIN_DIRS = {
    "environment": "src/env/",
    "agent": "src/agents/",
    "sensor": "src/plugins/",
    "actuator": "src/plugins/",
}


def scaffold_plugin(plugin_type, name):
    if plugin_type not in TEMPLATES:
        raise ValueError(f"Unknown plugin type: {plugin_type}")
    class_name = name[0].upper() + name[1:]
    code = TEMPLATES[plugin_type].format(
        name=class_name, date=datetime.now().strftime("%Y-%m-%d")
    )
    dir_path = PLUGIN_DIRS[plugin_type]
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{name.lower()}.py")
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
    with open(file_path, "w") as f:
        f.write(code)
    print(f"Created {plugin_type} plugin: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new plugin (environment, agent, sensor, actuator)"
    )
    parser.add_argument(
        "type", choices=TEMPLATES.keys(), help="Type of plugin to scaffold"
    )
    parser.add_argument("name", help="Name of the plugin class (CamelCase)")
    args = parser.parse_args()
    scaffold_plugin(args.type, args.name)


if __name__ == "__main__":
    main()
