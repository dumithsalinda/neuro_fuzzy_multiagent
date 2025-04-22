import subprocess
import os
from typing import List, Tuple

class PluginLinter:
    """
    Runs flake8 linter on plugin files to enforce code quality and style.
    """
    def __init__(self, flake8_path: str = "flake8"):
        self.flake8_path = flake8_path

    def lint_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Lint a single file. Returns (passed, [list of error lines])
        """
        if not os.path.exists(file_path):
            return False, [f"File not found: {file_path}"]
        try:
            result = subprocess.run([
                self.flake8_path, file_path, "--format=%(row)d:%(col)d: %(code)s %(text)s"
            ], capture_output=True, text=True)
            errors = result.stdout.strip().split("\n") if result.stdout.strip() else []
            passed = (result.returncode == 0)
            return passed, errors
        except Exception as e:
            return False, [f"Linting failed: {e}"]

    def lint_files(self, file_paths: List[str]) -> Tuple[bool, List[str]]:
        """
        Lint multiple files. Returns (all_passed, [all error lines])
        """
        all_passed = True
        all_errors = []
        for fp in file_paths:
            passed, errors = self.lint_file(fp)
            if not passed:
                all_passed = False
            all_errors.extend([f"{os.path.basename(fp)}: {e}" for e in errors])
        return all_passed, all_errors
