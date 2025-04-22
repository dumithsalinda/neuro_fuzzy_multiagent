"""
plugin_validation.py

Automated checks for plugin submissions.
"""
import tempfile
import os
import ast
import importlib.util
import requests
from typing import Dict, Any
import logging

def dynamic_import_from_code(code: str, plugin_name: str) -> Any:
    """
    Dynamically import a module from source code.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, f"{plugin_name}.py")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, fname)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise Exception(f"Failed to dynamically import {plugin_name}: {e}")

def validate_plugin_from_url(source_url: str, plugin_type: str, plugin_name: str) -> Dict[str, Any]:
    """
    Validate a plugin by fetching its source code from a URL, checking syntax and basic type compliance.
    Returns a report dict with 'passed', 'issues', and 'warnings'.
    """
    report = {"passed": True, "issues": [], "warnings": []}
    try:
        r = requests.get(source_url, timeout=10)
        if r.status_code != 200:
            report["passed"] = False
            report["issues"].append(f"Failed to fetch source file: {r.status_code}")
            logging.error(f"Failed to fetch plugin source: {source_url} (status {r.status_code})")
            return report
        code = r.text
    except Exception as e:
        report["passed"] = False
        report["issues"].append(f"Exception fetching source: {e}")
        logging.error(f"Exception fetching plugin source from {source_url}: {e}")
        return report
    # Syntax check
    try:
        ast.parse(code)
    except Exception as e:
        report["passed"] = False
        report["issues"].append(f"Syntax error: {e}")
        logging.error(f"Syntax error in plugin {plugin_name}: {e}")
        return report
    # Dynamic import and checks
    try:
        mod = dynamic_import_from_code(code, plugin_name)
        # Attribute checks
        if not hasattr(mod, "__version__"):
            report["warnings"].append("No __version__ attribute found.")
        if not mod.__doc__:
            report["warnings"].append("No module docstring found.")
        # Type-specific checks (basic)
        if plugin_type == "environment":
            found = any("BaseEnvironment" in b.__name__ for n, b in vars(mod).items() if isinstance(b, type))
            if not found:
                report["warnings"].append("No class inheriting from BaseEnvironment found.")
        if plugin_type == "agent":
            found = any("BaseAgent" in b.__name__ for n, b in vars(mod).items() if isinstance(b, type))
            if not found:
                report["warnings"].append("No class inheriting from BaseAgent found.")
    except Exception as e:
        logging.error(f"Error during dynamic import or checks for plugin {plugin_name}: {e}")
        report["warnings"].append(f"Error during dynamic import or checks: {e}")
    return report
