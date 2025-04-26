import hashlib
import json
import os
from typing import Dict, List, Optional

from neuro_fuzzy_multiagent.utils.model_schema import validate_model_metadata


class ModelRegistry:
    """
    Simple file-based model registry for registering, listing, and querying models.
    Each model must have a directory with a model.json metadata file.
    """

    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def register_model(self, model_dir: str) -> bool:
        """Register a model by validating and copying its metadata to the registry index."""
        meta_path = os.path.join(model_dir, "model.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No model.json found in {model_dir}")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        # Validate schema
        validate_model_metadata(metadata)
        name = metadata.get("name")
        if not name:
            raise ValueError("model.json must have a 'name' field")
        # Hash check (if model file exists)
        model_file = None
        for ext in ["onnx", "pt", "pth", "h5"]:
            candidate = os.path.join(model_dir, f"model.{ext}")
            if os.path.exists(candidate):
                model_file = candidate
                break
        if model_file:
            with open(model_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if "hash" in metadata and not metadata["hash"].endswith(file_hash):
                raise ValueError(
                    f"Model file hash does not match metadata. Got {file_hash}"
                )
        # Signature verification (if present)
        if "signature" in metadata:
            from neuro_fuzzy_multiagent.utils.crypto_utils import (
                load_public_key,
                verify_signature,
            )

            pubkey_path = os.environ.get("NFMAOS_MODEL_PUBKEY", None)
            if not pubkey_path or not os.path.exists(pubkey_path):
                raise ValueError(
                    "Public key for signature verification not found. Set NFMAOS_MODEL_PUBKEY env variable."
                )
            public_key = load_public_key(pubkey_path)
            # The signature is over the hash field (or model file contents)
            data_to_verify = metadata["hash"].encode()
            if not verify_signature(data_to_verify, metadata["signature"], public_key):
                raise ValueError("Model signature verification failed.")
        # Index file for registry
        index_path = os.path.join(self.registry_dir, f"{name}.json")
        with open(index_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return True

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return [f[:-5] for f in os.listdir(self.registry_dir) if f.endswith(".json")]

    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a registered model by name."""
        index_path = os.path.join(self.registry_dir, f"{model_name}.json")
        if not os.path.exists(index_path):
            return None
        with open(index_path, "r") as f:
            return json.load(f)


# Example usage and test
def test_model_registry():
    import shutil
    import tempfile

    # Setup temp dirs
    temp_registry = tempfile.mkdtemp()
    temp_model = tempfile.mkdtemp()
    try:
        # Create dummy model.json
        meta = {
            "name": "test_model",
            "version": "1.0.0",
            "author": "Test",
            "description": "Test model.",
            "supported_device": "test_device",
            "input_schema": "float32[1, 2]",
            "output_schema": "float32[1]",
            "hash": "sha256:dummy",
            "model_type": "cnn",
            "framework": "onnx",
        }
        with open(os.path.join(temp_model, "model.json"), "w") as f:
            json.dump(meta, f)
        registry = ModelRegistry(temp_registry)
        registry.register_model(temp_model)
        print("Registered models:", registry.list_models())
        print("Metadata:", registry.get_model_metadata("test_model"))
    finally:
        shutil.rmtree(temp_registry)
        shutil.rmtree(temp_model)


if __name__ == "__main__":
    test_model_registry()
