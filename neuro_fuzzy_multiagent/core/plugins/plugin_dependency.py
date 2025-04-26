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
        Checks if all requirements are installed by attempting to import each package.
        Returns (bool, output)
        """
        if not self.has_requirements():
            return True, "No requirements.txt found."
        missing = []
        with open(self.requirements_path, "r") as f:
            for line in f:
                pkg = line.strip().split("==")[0].split(">=")[0].split("<=")[0]
                if not pkg or pkg.startswith("#"):
                    continue
                try:
                    __import__(pkg)
                except ImportError:
                    missing.append(pkg)
        if missing:
            return False, f"Missing packages: {', '.join(missing)}"
        return True, "All requirements satisfied."

    def install_requirements(self) -> tuple:
        """
        Installs requirements for the plugin in the current environment.
        Returns (bool, output)
        """
        if not self.has_requirements():
            return True, "No requirements.txt found."
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(self.requirements_path),
        ]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            logging.info(
                f"Installed requirements for {self.plugin_dir}: returncode={proc.returncode}"
            )
            return proc.returncode == 0, proc.stdout + proc.stderr
        except Exception as e:
            logging.error(f"Failed to install requirements for {self.plugin_dir}: {e}")
            return False, str(e)


# Example usage (for test):
if __name__ == "__main__":
    mgr = PluginDependencyManager("../../plugins")
    print(mgr.has_requirements())
    print(mgr.check_requirements_installed())
