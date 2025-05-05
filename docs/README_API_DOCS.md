# API Documentation (Sphinx)

This directory contains the configuration for auto-generating Python API documentation using Sphinx.

## How to Build the Docs

1. **Install Sphinx and extensions:**
   ```bash
   pip install sphinx sphinx-autodoc-typehints
   ```
2. **Build the documentation:**
   ```bash
   cd docs
   sphinx-apidoc -o . ../neuro_fuzzy_multiagent/core
   sphinx-build -b html . _build/html
   ```
3. **View the docs:**
   Open `_build/html/index.html` in your browser.

## Updating the Docs
- Re-run the above commands after code changes.
- Commit both the `docs/` folder and any updated API docs if sharing with others.

## Notes
- The API docs include all public classes and functions in the core plugin system.
- See `conf.py` for configuration and `index.rst` for the docs structure.
