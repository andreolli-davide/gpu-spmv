# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SpMV GPU Optimization"
copyright = "2026, SpMV Team"
author = "SpMV Team"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.mathjax"]

import os

try:
    import breathe

    extensions.append("breathe")
    breathe_xml_path = os.environ.get(
        "SPHINX_BREATHE_XML", "@CMAKE_CURRENT_BINARY_DIR@/xml"
    )
    breathe_projects = {"spmv": breathe_xml_path}
    breathe_default_project = "spmv"
except ImportError:
    pass

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
