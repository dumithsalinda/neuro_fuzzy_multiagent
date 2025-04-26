import os
from src.utils.model_registry import ModelRegistry
from src.utils.model_loader import ModelLoader

class ExampleAgent:
    def __init__(self, registry_dir: str, device_type: str):
        self.registry = ModelRegistry(registry_dir)
        self.device_type = device_type
        self.model_loader = None

    def find_and_load_model(self):
        # Find a compatible model for the device
        for model_name in self.registry.list_models():
            meta = self.registry.get_model_metadata(model_name)
            if meta and meta.get("supported_device") == self.device_type:
                model_dir = os.path.join("/opt/nfmaos/models", model_name)
                self.model_loader = ModelLoader(model_dir)
                print(f"Loaded model: {model_name}")
                return True
        print("No compatible model found.")
        return False

    def predict(self, input_data):
        if not self.model_loader:
            raise RuntimeError("No model loaded.")
        return self.model_loader.predict(input_data)

# Example test usage
def test_agent_with_model():
    import tempfile
    import shutil
    temp_registry = tempfile.mkdtemp()
    temp_model = tempfile.mkdtemp()
    try:
        # Create dummy model.json
        import json
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
            "framework": "onnx"
        }
        with open(os.path.join(temp_model, "model.json"), "w") as f:
            json.dump(meta, f)
        # Register model
        registry = ModelRegistry(temp_registry)
        registry.register_model(temp_model)
        # Simulate agent
        agent = ExampleAgent(temp_registry, "test_device")
        agent.find_and_load_model()
        # agent.predict(...)  # Would require a real model file
    finally:
        shutil.rmtree(temp_registry)
        shutil.rmtree(temp_model)

if __name__ == "__main__":
    test_agent_with_model()
