import os
import tempfile
import shutil
import time
import json
import threading
from src.utils.model_registry import ModelRegistry
from src.utils.model_loader import ModelLoader
from examples.agent_with_model_hot_reload import HotReloadAgent

import hashlib

def make_model_dir(base_dir, name, device, framework="onnx", model_type="cnn", signature=None):
    model_dir = os.path.join(base_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    # Use a minimal valid ONNX model file for testing
    minimal_onnx_path = os.path.join(os.path.dirname(__file__), "minimal_valid.onnx")
    model_file_path = os.path.join(model_dir, "model.onnx")
    with open(minimal_onnx_path, "rb") as src, open(model_file_path, "wb") as dst:
        dst.write(src.read())
    # Compute the actual hash
    with open(model_file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    meta = {
        "name": name,
        "version": "1.0.0",
        "author": "Test",
        "description": f"Test model for {device}",
        "supported_device": device,
        "input_schema": "float32[1,2]",
        "output_schema": "float32[1]",
        "hash": f"sha256:{file_hash}",
        "model_type": model_type,
        "framework": framework,
        "model_dir": model_dir  # Add this for agent lookup
    }
    if signature:
        meta["signature"] = signature
    with open(os.path.join(model_dir, "model.json"), "w") as f:
        json.dump(meta, f)
    return model_dir

def test_hot_reload_agent_workflow():
    temp_registry = tempfile.mkdtemp()
    temp_models = tempfile.mkdtemp()
    os.environ['NFMAOS_MODEL_REGISTRY'] = temp_registry
    device_type = "test_device"
    # Create and register initial model
    model1_dir = make_model_dir(temp_models, "model1", device_type)
    registry = ModelRegistry(temp_registry)
    registry.register_model(model1_dir)
    # Start agent in a thread
    agent = HotReloadAgent(temp_registry, device_type)
    agent_thread = threading.Thread(target=agent.start, daemon=True)
    agent_thread.start()
    time.sleep(2)
    # Register a new model (should trigger reload)
    model2_dir = make_model_dir(temp_models, "model2", device_type)
    registry.register_model(model2_dir)
    time.sleep(2)
    # Remove model2 (should trigger reload back to model1)
    os.remove(os.path.join(temp_registry, "model2.json"))
    time.sleep(2)
    # Clean up
    agent.watcher.stop()
    shutil.rmtree(temp_registry)
    shutil.rmtree(temp_models)
    print("Hot reload agent workflow test completed.")

if __name__ == "__main__":
    test_hot_reload_agent_workflow()
