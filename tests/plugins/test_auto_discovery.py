import os
from src.core.plugins.auto_discovery import PluginRegistry


def test_plugin_auto_discovery():
    # Use the plugins directory for discovery
    plugin_dir = os.path.dirname(__file__)
    # Simulate a plugin with __plugin_name__ and __version__
    with open(os.path.join(plugin_dir, "dummy_plugin.py"), "w") as f:
        f.write(
            """
__plugin_name__ = 'DummyPlugin'
__version__ = '1.2.3'
def hello():
    return 'hi'
"""
        )
    registry = PluginRegistry(plugin_dir=plugin_dir, base_package="tests.plugins")
    registry.discover_plugins()
    plugins = registry.list_plugins()
    assert any(
        name == "DummyPlugin" and version == "1.2.3" for name, version in plugins
    )
    dummy = registry.get_plugin("DummyPlugin")
    assert dummy.module.hello() == "hi"
    # Test hot-reload
    registry.reload_plugin("DummyPlugin")
    os.remove(os.path.join(plugin_dir, "dummy_plugin.py"))
