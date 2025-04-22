import importlib
import pkgutil
import os
from pathlib import Path

class PluginInfo:
    def __init__(self, name, version, module):
        self.name = name
        self.version = version
        self.module = module

import logging
from typing import Optional, Dict, Any

class PluginRegistry:
    """
    Auto-discovers plugins in the given directory/package, tracks version info, and supports hot-reloading.
    Plugins should define __version__ and __plugin_name__ attributes.
    """
    def __init__(self, plugin_dir: str, base_package: Optional[str] = None):
        """
        Args:
            plugin_dir (str): Directory to scan for plugins.
            base_package (str, optional): Base package name for import. Defaults to plugin_dir name.
        """
        self.plugin_dir = Path(plugin_dir)
        self.base_package = base_package or self.plugin_dir.name
        self.plugins: Dict[str, Any] = {}
        self.discover_plugins()

    def discover_plugins(self) -> None:
        """
        Discover and import all plugins in the plugin directory.
        Populates self.plugins with plugin info.
        """
        self.plugins.clear()
        for finder, name, ispkg in pkgutil.iter_modules([str(self.plugin_dir)]):
            try:
                logging.info(f"Discovering plugin: {name}")
                module = importlib.import_module(f"{self.base_package}.{name}")
                version = getattr(module, "__version__", "unknown")
                self.plugins[name] = PluginInfo(name, version, module)
            except Exception as e:
                logging.error(f"Failed to import plugin {name}: {e}")
                full_name = f"{self.base_package}.{name}"
                module = importlib.import_module(full_name)
                plugin_name = getattr(module, '__plugin_name__', name)
                version = getattr(module, '__version__', '0.1.0')
                self.plugins[plugin_name] = PluginInfo(plugin_name, version, module)
            except Exception as e:
                print(f"Failed to import plugin {name}: {e}")

    def get_plugin(self, name: str) -> Optional[Any]:
        """
        Get a plugin by name.
        """
        return self.plugins.get(name)

    def list_plugins(self) -> list:
        """
        List all discovered plugins as (name, version) tuples.
        """
        return [(p.name, p.version) for p in self.plugins.values()]

    def reload_plugin(self, name: str) -> bool:
        """
        Reload a specific plugin module by name.
        Returns True if successful, False otherwise.
        """
        info = self.plugins.get(name)
        if info:
            try:
                importlib.reload(info.module)
                # Update version in case it changed
                info.version = getattr(info.module, '__version__', info.version)
                logging.info(f"Reloaded plugin: {name}")
                return True
            except Exception as e:
                logging.error(f"Failed to reload plugin {name}: {e}")
        return False

    def reload_all(self) -> None:
        """
        Reload all discovered plugin modules.
        """
        for name in self.plugins:
            self.reload_plugin(name)

