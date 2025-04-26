import os
import time
import threading
from typing import Callable, Set

class RegistryWatcher:
    """
    Watches a directory for changes (new/removed/modified model files).
    Calls a callback when a change is detected.
    """
    def __init__(self, registry_dir: str, callback: Callable[[Set[str], Set[str], Set[str]], None], poll_interval: float = 2.0):
        self.registry_dir = registry_dir
        self.callback = callback
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._last_seen = set()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _watch_loop(self):
        while not self._stop_event.is_set():
            current = set(f for f in os.listdir(self.registry_dir) if f.endswith('.json'))
            added = current - self._last_seen
            removed = self._last_seen - current
            modified = set()
            for f in current & self._last_seen:
                path = os.path.join(self.registry_dir, f)
                if os.path.getmtime(path) > getattr(self, f'_mtime_{f}', 0):
                    modified.add(f)
                    setattr(self, f'_mtime_{f}', os.path.getmtime(path))
            if added or removed or modified:
                self.callback(added, removed, modified)
            self._last_seen = current
            time.sleep(self.poll_interval)

# Example usage in an agent:
def example_callback(added, removed, modified):
    if added:
        print(f"Models added: {added}")
    if removed:
        print(f"Models removed: {removed}")
    if modified:
        print(f"Models modified: {modified}")

if __name__ == "__main__":
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    try:
        watcher = RegistryWatcher(temp_dir, example_callback, poll_interval=1.0)
        watcher.start()
        # Simulate changes
        import time
        with open(os.path.join(temp_dir, 'foo.json'), 'w') as f:
            f.write('{}')
        time.sleep(2)
        with open(os.path.join(temp_dir, 'foo.json'), 'w') as f:
            f.write('{"changed": true}')
        time.sleep(2)
        os.remove(os.path.join(temp_dir, 'foo.json'))
        time.sleep(2)
        watcher.stop()
    finally:
        shutil.rmtree(temp_dir)
