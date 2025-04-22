import os
import subprocess
import sys
import logging
from pathlib import Path
from src.core.plugins.plugin_linter import PluginLinter

class PluginDependencyManager:
    """
    Handles detection, validation, and installation of plugin-specific dependencies.
    Each plugin may include a requirements.txt file in its directory.
    """
    def __init__(self, plugin_dir):
        self.plugin_dir = Path(plugin_dir)
        self.requirements_path = self.plugin_dir / "requirements.txt"

    def has_requirements(self):
        return self.requirements_path.exists()

    def check_requirements_installed(self):
        """
        Checks if all requirements are installed by attempting a dry-run install.
        Returns (bool, output)
        """
        if not self.has_requirements():
            return True, "No requirements.txt found."
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "-r", str(self.requirements_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # If 'would install' is present, something is missing
        needs_install = "Would install" in proc.stdout or proc.returncode != 0
        return not needs_install, proc.stdout + proc.stderr

    def install_requirements(self):
        """
        Installs requirements for the plugin in the current environment.
        Returns (bool, output)
        """
        if not self.has_requirements():
            return True, "No requirements.txt found."
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode == 0, proc.stdout + proc.stderr

# Example usage (for test):
if __name__ == "__main__":
    mgr = PluginDependencyManager("../../plugins")
    print(mgr.has_requirements())
    print(mgr.check_requirements_installed())
