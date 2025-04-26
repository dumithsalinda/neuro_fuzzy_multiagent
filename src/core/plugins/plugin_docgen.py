import os
import importlib.util
import inspect
from typing import List


def get_plugin_classes(module_path: str) -> List[type]:
    """Dynamically load all classes from a module file."""
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        return []
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return []  # Skip broken plugins
    classes = [
        obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module_name
    ]
    return classes


def generate_plugin_docs(plugin_dir: str, output_md: str):
    """Generate Markdown documentation for all plugin classes in a directory."""
    lines = [f"# Plugin Documentation for `{plugin_dir}`\n"]
    for fname in sorted(os.listdir(plugin_dir)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        fpath = os.path.join(plugin_dir, fname)
        classes = get_plugin_classes(fpath)
        for cls in classes:
            lines.append(f"## `{cls.__name__}`\n")
            doc = inspect.getdoc(cls) or "No docstring provided."
            lines.append(doc + "\n")
            # List methods
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith("_"):
                    continue
                sig = str(inspect.signature(method))
                mdoc = inspect.getdoc(method) or ""
                lines.append(f"### `{name}{sig}`\n{mdoc}\n")
    with open(output_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Plugin docs written to {output_md}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Markdown docs for plugins.")
    parser.add_argument(
        "--plugin-dir", required=True, help="Directory with plugin .py files"
    )
    parser.add_argument("--output", required=True, help="Output Markdown file")
    args = parser.parse_args()
    generate_plugin_docs(args.plugin_dir, args.output)
