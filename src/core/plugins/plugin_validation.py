"""
plugin_validation.py

Automated checks for plugin submissions.
"""
import tempfile
import os
import ast
import importlib.util
import requests

def validate_plugin_from_url(source_url, plugin_type, plugin_name):
    report = {"passed": True, "issues": [], "warnings": []}
    try:
        r = requests.get(source_url, timeout=10)
        if r.status_code != 200:
            report["passed"] = False
            report["issues"].append(f"Failed to fetch source file: {r.status_code}")
            return report
        code = r.text
    except Exception as e:
        report["passed"] = False
        report["issues"].append(f"Exception fetching source: {e}")
        return report
    # Syntax check
    try:
        ast.parse(code)
    except Exception as e:
        report["passed"] = False
        report["issues"].append(f"Syntax error: {e}")
        return report
    # Save to temp file and try to import
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, f"{plugin_name}.py")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, fname)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            report["passed"] = False
            report["issues"].append(f"Import error: {e}")
            return report
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
    return report
