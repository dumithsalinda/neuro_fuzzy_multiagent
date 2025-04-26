#!/usr/bin/env python3
import argparse
import os
import sys

from neuro_fuzzy_multiagent.utils.model_registry import ModelRegistry

REGISTRY_DIR = os.environ.get("NFMAOS_MODEL_REGISTRY", "/opt/nfmaos/registry")


def main():
    parser = argparse.ArgumentParser(description="NFMA-OS Model Registry CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register
    p_register = subparsers.add_parser(
        "register", help="Register a new model directory"
    )
    p_register.add_argument("model_dir", help="Path to directory with model.json")

    # List
    p_list = subparsers.add_parser("list", help="List registered models")

    # Inspect
    p_inspect = subparsers.add_parser("inspect", help="Show metadata for a model")
    p_inspect.add_argument("model_name", help="Name of the model to inspect")

    # Remove
    p_remove = subparsers.add_parser("remove", help="Remove a model from the registry")
    p_remove.add_argument("model_name", help="Name of the model to remove")

    args = parser.parse_args()
    registry = ModelRegistry(REGISTRY_DIR)

    if args.command == "register":
        try:
            registry.register_model(args.model_dir)
            print(f"Model registered from {args.model_dir}")
        except Exception as e:
            print(f"Registration failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "list":
        models = registry.list_models()
        print("Registered models:")
        for m in models:
            print(f"  {m}")
    elif args.command == "inspect":
        meta = registry.get_model_metadata(args.model_name)
        if not meta:
            print(f"Model '{args.model_name}' not found.", file=sys.stderr)
            sys.exit(1)
        import json

        print(json.dumps(meta, indent=2))
    elif args.command == "remove":
        import os

        meta_path = os.path.join(REGISTRY_DIR, f"{args.model_name}.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
            print(f"Removed model: {args.model_name}")
        else:
            print(f"Model '{args.model_name}' not found.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
