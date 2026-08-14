import os
import sys

# Point Sphinx to the root directory containing your source code
sys.path.insert(0, os.path.abspath(".."))

# Specify project metadata
project = "OIC Toolkit"
copyright = "2026"
author = "Jian Wei Tay"

# Register Sphinx extensions
extensions = [
    "sphinx_gallery.gen_gallery",  # Executes and builds example scripts
    "autoapi.extension",  # Generates API reference docs from source
    "sphinx.ext.napoleon",  # Parses NumPy-style docstrings
]

# Configure AutoAPI
autoapi_dirs = ["../src/oic_toolkit"]  # Points to your module source code
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_ignore = ["*_internal.py"]

# Configure Sphinx Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # Path to raw Python example scripts
    "gallery_dirs": "auto_examples",  # Target path where generated HTML is written
}

# Configure Theme
html_theme = "pydata_sphinx_theme"
