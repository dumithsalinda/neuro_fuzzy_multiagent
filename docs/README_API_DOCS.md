# API Documentation (Sphinx)

> **Note:** For all code documentation standards and plugin doc generation, see the [Developer Guide](DEVELOPER.md).

This directory contains the configuration for auto-generating Python API documentation using Sphinx.

## Quick Build Steps

1. Install Sphinx and extensions:
   ```bash
   pip install sphinx sphinx-autodoc-typehints
   ```
2. Build the documentation:
   ```bash
   cd docs
   sphinx-apidoc -o . ../neuro_fuzzy_multiagent/core
   sphinx-build -b html . _build/html
   ```
3. Open `_build/html/index.html` in your browser.

For updating or customizing API docs, see the [Developer Guide](DEVELOPER.md).
