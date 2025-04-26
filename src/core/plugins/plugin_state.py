import json
import pickle
from pathlib import Path


class PluginStateManager:
    """
    Handles saving and loading plugin state to disk (JSON or pickle).
    Plugins can use this to persist state across reloads or restarts.
    """

    def __init__(self, plugin_dir, state_filename="state.json"):
        self.plugin_dir = Path(plugin_dir)
        self.state_path = self.plugin_dir / state_filename

    def save_json(self, state_dict):
        with open(self.state_path, "w") as f:
            json.dump(state_dict, f)

    def load_json(self):
        if not self.state_path.exists():
            return None
        with open(self.state_path, "r") as f:
            return json.load(f)

    def save_pickle(self, obj, filename="state.pkl"):
        pkl_path = self.plugin_dir / filename
        with open(pkl_path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, filename="state.pkl"):
        pkl_path = self.plugin_dir / filename
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)


# Example usage (for test):
if __name__ == "__main__":
    mgr = PluginStateManager("../../plugins")
    mgr.save_json({"foo": 42})
    print(mgr.load_json())
    mgr.save_pickle([1, 2, 3])
    print(mgr.load_pickle())
