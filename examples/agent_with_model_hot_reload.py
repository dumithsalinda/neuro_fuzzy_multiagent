import os
import time

from neuro_fuzzy_multiagent.utils.model_loader import ModelLoader
from neuro_fuzzy_multiagent.utils.model_registry import ModelRegistry
from neuro_fuzzy_multiagent.utils.registry_watcher import RegistryWatcher


class HotReloadAgent:
    def __init__(self, registry_dir: str, device_type: str):
        self.registry = ModelRegistry(registry_dir)
        self.device_type = device_type
        self.model_loader = None
        self.current_model_name = None
        self.registry_dir = registry_dir
        self.watcher = RegistryWatcher(registry_dir, self.on_registry_change)

    def start(self):
        self.watcher.start()
        self.find_and_load_model()
        print("Agent started. Watching for model changes...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down agent...")
            self.watcher.stop()

    def find_and_load_model(self):
        for model_name in self.registry.list_models():
            meta = self.registry.get_model_metadata(model_name)
            if meta and meta.get("supported_device") == self.device_type:
                model_dir = meta.get(
                    "model_dir", os.path.join("/opt/nfmaos/models", model_name)
                )
                if self.current_model_name != model_name:
                    self.model_loader = ModelLoader(model_dir)
                    self.current_model_name = model_name
                    print(f"Loaded model: {model_name} from {model_dir}")
                return True
        print("No compatible model found.")
        self.model_loader = None
        self.current_model_name = None
        return False

    def on_registry_change(self, added, removed, modified):
        print(
            f"Registry changed: added={added}, removed={removed}, modified={modified}"
        )
        # Reload model if a compatible one was added, removed, or modified
        self.find_and_load_model()

    def predict(self, input_data):
        if not self.model_loader:
            raise RuntimeError("No model loaded.")
        return self.model_loader.predict(input_data)


if __name__ == "__main__":
    # Example usage: simulate agent for 'test_device'
    registry_dir = os.environ.get("NFMAOS_MODEL_REGISTRY", "/opt/nfmaos/registry")
    agent = HotReloadAgent(registry_dir, "test_device")
    agent.start()
