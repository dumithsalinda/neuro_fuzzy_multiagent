import tempfile
import os
from src.core.plugins.plugin_signature import PluginSignatureVerifier


def test_signature_verification_valid():
    # Create a dummy plugin file
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_path = os.path.join(tmpdir, "plugin1.py")
        with open(plugin_path, "w") as f:
            f.write("print('hello world')\n")
        # Compute hash
        verifier = PluginSignatureVerifier(
            manifest_path=os.path.join(tmpdir, "manifest.txt")
        )
        hashval = verifier.compute_hash(plugin_path)
        # Write manifest
        with open(os.path.join(tmpdir, "manifest.txt"), "w") as f:
            f.write(f"plugin1.py: {hashval}\n")
        verifier = PluginSignatureVerifier(
            manifest_path=os.path.join(tmpdir, "manifest.txt")
        )
        assert verifier.verify(plugin_path)


def test_signature_verification_tampered():
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_path = os.path.join(tmpdir, "plugin2.py")
        with open(plugin_path, "w") as f:
            f.write("print('hello world')\n")
        verifier = PluginSignatureVerifier(
            manifest_path=os.path.join(tmpdir, "manifest.txt")
        )
        hashval = verifier.compute_hash(plugin_path)
        with open(os.path.join(tmpdir, "manifest.txt"), "w") as f:
            f.write(f"plugin2.py: {hashval}\n")
        # Tamper with file
        with open(plugin_path, "a") as f:
            f.write("# tampered\n")
        verifier = PluginSignatureVerifier(
            manifest_path=os.path.join(tmpdir, "manifest.txt")
        )
        try:
            verifier.verify(plugin_path)
            assert False, "Should have raised ValueError for tampered file"
        except ValueError as e:
            assert "Signature mismatch" in str(e)


def test_signature_verification_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_path = os.path.join(tmpdir, "plugin3.py")
        with open(plugin_path, "w") as f:
            f.write("print('hello world')\n")
        verifier = PluginSignatureVerifier(
            manifest_path=os.path.join(tmpdir, "manifest.txt")
        )
        try:
            verifier.verify(plugin_path)
            assert False, "Should have raised ValueError for missing signature"
        except ValueError as e:
            assert "No signature found" in str(e)
