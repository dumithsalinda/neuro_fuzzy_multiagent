# Sphinx configuration for API documentation
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Neuro-Fuzzy Multi-Agent System'
copyright = '2025, Neuro-Fuzzy Team'
author = 'Neuro-Fuzzy Team'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
