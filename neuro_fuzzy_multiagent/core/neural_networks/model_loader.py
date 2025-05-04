"""
model_loader.py
Plug-and-play loader for pretrained AI models (PyTorch, Keras, ONNX, or custom formats).
Allows dynamic installation and registration of new AI drivers.
"""
import os
import importlib

MODEL_REGISTRY = {}

# Optional: import torch/keras/onnx only if available
try:
    import torch
except ImportError:
    torch = None
try:
    from tensorflow import keras
except ImportError:
    keras = None
try:
    import onnx
except ImportError:
    onnx = None

def register_model(name, model):
    """Register a loaded model for plug-and-play use."""
    MODEL_REGISTRY[name] = model
    return model

def get_registered_models():
    return dict(MODEL_REGISTRY)

def load_pretrained_model(model_path, model_type, name=None, custom_loader=None, **kwargs):
    """
    Loads and registers a pretrained model for plug-and-play use.
    Args:
        model_path (str): Path to the model file or directory.
        model_type (str): One of 'torch', 'keras', 'onnx', or 'custom'.
        name (str): Optional name for registration (defaults to filename).
        custom_loader (callable): Optional loader for custom model types.
        **kwargs: Extra args for loader.
    Returns:
        The loaded model instance.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(model_path))[0]
    if model_type == 'torch':
        if torch is None:
            raise ImportError("PyTorch not installed!")
        model = torch.load(model_path, **kwargs)
    elif model_type == 'keras':
        if keras is None:
            raise ImportError("TensorFlow/Keras not installed!")
        model = keras.models.load_model(model_path, **kwargs)
    elif model_type == 'onnx':
        if onnx is None:
            raise ImportError("ONNX not installed!")
        model = onnx.load(model_path, **kwargs)
    elif model_type == 'custom':
        if not custom_loader:
            raise ValueError("Custom model_type requires a custom_loader function!")
        model = custom_loader(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    register_model(name, model)
    return model

# Example usage:
# model = load_pretrained_model('my_model.pt', 'torch', name='my_torch_model')
# model = load_pretrained_model('my_model.h5', 'keras', name='my_keras_model')
# model = load_pretrained_model('my_model.onnx', 'onnx', name='my_onnx_model')
# model = load_pretrained_model('my_custom.bin', 'custom', custom_loader=my_loader_fn)

# To retrieve a model:
# model = get_registered_models()['my_torch_model']
