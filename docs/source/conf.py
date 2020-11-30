# Configuration file for the Sphinx documentation builder.
#
# To auto-generate rst files for the program use the command:
# sphinx-apidoc -f -o ./source/tmglow/ ../tmglow/
# Use "make clean" and "make html" to build the docs

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../tmglow'))

from sphinx.ext.autodoc import between

def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word ======
    app.connect('autodoc-process-docstring', between('^.*=====.*$', exclude=True))
    return app

# -- Project information -----------------------------------------------------

project = 'Multi-fidelity Generative Deep Learning Turbulent Flows'
copyright = '2020, Nicholas Geneva'
author = 'Nicholas Geneva'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# Change homepage file to index
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    'sphinx.ext.todo',
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    'sphinx.ext.ifconfig'
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']