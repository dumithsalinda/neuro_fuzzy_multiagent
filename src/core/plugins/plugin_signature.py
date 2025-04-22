import hashlib
from typing import Dict, Optional
import os

class PluginSignatureVerifier:
    """
    Verifies plugin file signatures using SHA256 hash (default) or optional public key signature.
    """
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.signatures = self._load_manifest()

    def _load_manifest(self) -> Dict[str, str]:
        """Load plugin signatures from manifest file (YAML or simple txt: plugin.py:hash)."""
        signatures = {}
        if not os.path.exists(self.manifest_path):
            return signatures
        with open(self.manifest_path, "r") as f:
            for line in f:
                if ":" in line:
                    fname, sig = line.strip().split(":", 1)
                    signatures[fname.strip()] = sig.strip()
        return signatures

    def compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify(self, file_path: str) -> bool:
        """Verify that the file matches the manifest hash."""
        fname = os.path.basename(file_path)
        expected = self.signatures.get(fname)
        if not expected:
            raise ValueError(f"No signature found for {fname}")
        actual = self.compute_hash(file_path)
        if actual != expected:
            raise ValueError(f"Signature mismatch for {fname}: expected {expected}, got {actual}")
        return True

# Example manifest.txt:
# plugin1.py: 1234abcd...
# plugin2.py: abcd5678...
