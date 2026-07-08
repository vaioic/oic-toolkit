import os
import sys

# Step 2a: Point Sphinx to the root directory containing your python modules
sys.path.insert(0, os.path.abspath(".."))

project = "OIC Toolkit"
copyright = "2026"
author = "Jian Wei Tay"

# Step 2b: Register extensions
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",  # Parses NumPy-style docstrings
    "sphinx_gallery.gen_gallery",  # Executes examples and builds the gallery webpage
]

# Configure AutoAPI pathing rules
autoapi_dirs = ["../src/oic_toolkit"]  # Points to your module source code
autoapi_type = "python"

# Step 2c: Configure the Theme
html_theme = "pydata_sphinx_theme"

# Step 2d: Configure Sphinx-Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # Where your raw python example scripts live
    "gallery_dirs": "auto_examples",  # Where Sphinx will put the generated HTML outputs
    "image_scrapers": ("matplotlib",),
}

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    # "private-members",
]

autoapi_ignore = ["*_internal.py"]
