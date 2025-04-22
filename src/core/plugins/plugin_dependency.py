import os
import subprocess
import sys
import logging
from pathlib import Path


class PluginDependencyManager:
    """
    Handles detection, validation, and installation of plugin-specific dependencies.
    Each plugin may include a requirements.txt file in its directory.
    """
    def __init__(self, plugin_dir: str):
        """
        Args:
            plugin_dir (str): Path to the plugin directory.
        """
        self.plugin_dir = Path(plugin_dir)
        self.requirements_path = self.plugin_dir / "requirements.txt"

    def has_requirements(self) -> bool:
        """
        Returns True if the plugin has a requirements.txt file.
        """
        return self.requirements_path.exists()

    def check_requirements_installed(self) -> tuple:
        """
        Checks if all requirements are installed by attempting a dry-run install.
        Returns (bool, output)
        """
        if not self.has_requirements():
            return True, "No requirements.txt found."
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "-r", str(self.requirements_path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            needs_install = any(line.startswith("Would install") for line in proc.stdout.splitlines())
            logging.info(f"Checked requirements for {self.plugin_dir}: needs_install={needs_install}")
            return not needs_install, proc.stdout + proc.stderr
        except Exception as e:
            logging.error(f"Failed to check requirements for {self.plugin_dir}: {e}")
            return False, str(e)

    def install_requirements(self) -> tuple:
        """
        Installs requirements for the plugin in the current environment.
        Returns (bool, output)
        """
        if not self.has_requirements():
            return True, "No requirements.txt found."
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logging.info(f"Installed requirements for {self.plugin_dir}: returncode={proc.returncode}")
            return proc.returncode == 0, proc.stdout + proc.stderr
        except Exception as e:
            logging.error(f"Failed to install requirements for {self.plugin_dir}: {e}")
            return False, str(e)

# Example usage (for test):
if __name__ == "__main__":
    mgr = PluginDependencyManager("../../plugins")
    print(mgr.has_requirements())
    print(mgr.check_requirements_installed())
