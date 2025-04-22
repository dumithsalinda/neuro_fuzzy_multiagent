import tempfile
import os
import textwrap
from src.core.plugins.plugin_docgen import generate_plugin_docs

def test_generate_plugin_docs_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample plugin file
        plugin_code = textwrap.dedent('''
            class MyPlugin:
                """This is a test plugin."""
                def foo(self, x):
                    """Foo does something."""
                    return x
        ''')
        plugin_path = os.path.join(tmpdir, "my_plugin.py")
        with open(plugin_path, "w") as f:
            f.write(plugin_code)
        output_md = os.path.join(tmpdir, "docs.md")
        generate_plugin_docs(tmpdir, output_md)
        with open(output_md) as f:
            content = f.read()
            assert "MyPlugin" in content
            assert "This is a test plugin" in content
            assert "foo(self, x)" in content
            assert "Foo does something" in content
