"""
generate_plugin_docs.py

Auto-generates markdown documentation for all registered plugins (environments, agents, sensors, actuators).
"""
import inspect
from src.core.plugins.registration_utils import get_registered_plugins

PLUGIN_TYPES = ['environment', 'agent', 'sensor', 'actuator']


def document_class(cls):
    doc = f"### `{cls.__name__}`\n"
    if cls.__doc__:
        doc += f"{inspect.cleandoc(cls.__doc__)}\n\n"
    # Constructor signature
    try:
        sig = inspect.signature(cls.__init__)
        doc += f"**Constructor:** `{cls.__name__}{sig}`\n\n"
    except Exception:
        pass
    # List public methods (excluding dunder and base ABC methods)
    methods = [
        (name, meth)
        for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    if methods:
        doc += "**Methods:**\n"
        for name, meth in methods:
            doc += f"- `{name}{inspect.signature(meth)}`\n"
            if meth.__doc__:
                doc += f"    - {inspect.cleandoc(meth.__doc__)}\n"
    return doc + "\n"


def generate_all_plugin_docs():
    md = "# Plugin API Documentation\n\n"
    for ptype in PLUGIN_TYPES:
        md += f"## {ptype.capitalize()} Plugins\n\n"
        plugins = get_registered_plugins(ptype)
        if not plugins:
            md += "_No plugins registered._\n\n"
            continue
        for name, cls in plugins.items():
            md += document_class(cls)
    return md

if __name__ == "__main__":
    docs = generate_all_plugin_docs()
    with open("PLUGIN_DOCS.md", "w") as f:
        f.write(docs)
    print("Plugin documentation generated in PLUGIN_DOCS.md")
