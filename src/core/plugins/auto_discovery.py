import importlib
import pkgutil
import os
from pathlib import Path

class PluginInfo:
    def __init__(self, name, version, module):
        self.name = name
        self.version = version
        self.module = module

class PluginRegistry:
    """
    Auto-discovers plugins in the given directory/package, tracks version info, and supports hot-reloading.
    Plugins should define __version__ and __plugin_name__ attributes.
    """
    def __init__(self, plugin_dir, base_package=None):
        self.plugin_dir = Path(plugin_dir)
        self.base_package = base_package or self.plugin_dir.name
        self.plugins = {}
        self.discover_plugins()

    def discover_plugins(self):
        self.plugins.clear()
        for finder, name, ispkg in pkgutil.iter_modules([str(self.plugin_dir)]):
            try:
                full_name = f"{self.base_package}.{name}"
                module = importlib.import_module(full_name)
                plugin_name = getattr(module, '__plugin_name__', name)
                version = getattr(module, '__version__', '0.1.0')
                self.plugins[plugin_name] = PluginInfo(plugin_name, version, module)
            except Exception as e:
                print(f"Failed to import plugin {name}: {e}")

    def get_plugin(self, name):
        return self.plugins.get(name)

    def list_plugins(self):
        return [(p.name, p.version) for p in self.plugins.values()]

    def reload_plugin(self, name):
        info = self.plugins.get(name)
        if info:
            importlib.reload(info.module)
            # Update version in case it changed
            info.version = getattr(info.module, '__version__', info.version)
            return True
        return False

    def reload_all(self):
        for name in self.plugins:
            self.reload_plugin(name)
