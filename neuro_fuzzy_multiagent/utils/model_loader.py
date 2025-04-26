import importlib
import json
import os
from typing import Any, Dict

# Optional: Import AI runtimes if available
try:
    import onnxruntime
except ImportError:
    onnxruntime = None
try:
    import torch
except ImportError:
    torch = None
try:
    import tensorflow as tf
except ImportError:
    tf = None


class ModelLoaderException(Exception):
    pass


class ModelLoader:
    """
    Universal loader for different neural network types (neuro-fuzzy, CNN, RNN, transformer, etc.).
    Loads models based on metadata and exposes a standard predict interface.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        self.runtime = None
        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        from neuro_fuzzy_multiagent.utils.model_schema import validate_model_metadata

        meta_path = os.path.join(self.model_dir, "model.json")
        if not os.path.exists(meta_path):
            raise ModelLoaderException(f"Metadata file not found: {meta_path}")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        try:
            validate_model_metadata(self.metadata)
        except Exception as e:
            raise ModelLoaderException(f"Invalid model metadata: {e}")
        if "model_type" not in self.metadata or "framework" not in self.metadata:
            raise ModelLoaderException(
                "model.json must contain 'model_type' and 'framework'"
            )

    def _load_model(self):
        model_type = self.metadata["model_type"].lower()
        framework = self.metadata["framework"].lower()
        model_file = None
        # Find the first file with a known extension
        for ext in ["onnx", "pt", "pth", "h5"]:
            candidate = os.path.join(self.model_dir, f"model.{ext}")
            if os.path.exists(candidate):
                model_file = candidate
                break
        if not model_file:
            raise ModelLoaderException(
                f"No supported model file found in {self.model_dir}"
            )

        if framework == "onnx":
            if onnxruntime is None:
                raise ModelLoaderException("onnxruntime is not installed")
            self.model = onnxruntime.InferenceSession(model_file)
            self.runtime = "onnx"
        elif framework in ("torch", "pytorch"):
            if torch is None:
                raise ModelLoaderException("PyTorch is not installed")
            self.model = torch.load(model_file, map_location="cpu")
            self.model.eval()
            self.runtime = "torch"
        elif framework in ("tf", "tensorflow"):
            if tf is None:
                raise ModelLoaderException("TensorFlow is not installed")
            self.model = tf.keras.models.load_model(model_file)
            self.runtime = "tf"
        elif framework == "scikit-fuzzy":
            # Example: load a neuro-fuzzy model (custom implementation)
            # This is a placeholder; actual loading will depend on your neuro-fuzzy implementation
            module = importlib.import_module("scikitfuzzy")
            self.model = module.load(model_file)
            self.runtime = "scikit-fuzzy"
        else:
            raise ModelLoaderException(f"Unsupported framework: {framework}")

    def predict(self, input_data: Any) -> Any:
        """Run inference using the loaded model. Input/output format depends on the model type."""
        if self.runtime == "onnx":
            input_name = self.model.get_inputs()[0].name
            return self.model.run(None, {input_name: input_data})
        elif self.runtime == "torch":
            with torch.no_grad():
                return self.model(input_data)
        elif self.runtime == "tf":
            return self.model.predict(input_data)
        elif self.runtime == "scikit-fuzzy":
            return self.model.infer(input_data)
        else:
            raise ModelLoaderException("No valid runtime loaded.")

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


# Example test function
def test_model_loader():
    """Test loading and running a model (mocked for demonstration)."""
    import shutil
    import tempfile

    # Create a mock model directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Write dummy metadata
        meta = {
            "name": "dummy_model",
            "version": "1.0.0",
            "author": "Test",
            "description": "Dummy model for testing.",
            "supported_device": "test_device",
            "input_schema": "float32[1, 2]",
            "output_schema": "float32[1]",
            "hash": "sha256:dummy",
            "model_type": "cnn",
            "framework": "onnx",
        }
        with open(os.path.join(temp_dir, "model.json"), "w") as f:
            json.dump(meta, f)
        # Create a dummy ONNX file
        with open(os.path.join(temp_dir, "model.onnx"), "wb") as f:
            f.write(b"dummy")
        # Try to load (will fail unless onnxruntime is installed and file is valid)
        try:
            loader = ModelLoader(temp_dir)
            print("Loaded model metadata:", loader.get_metadata())
            # loader.predict(...)  # Would require a real model
        except Exception as e:
            print("Expected error (no real model):", e)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_model_loader()
