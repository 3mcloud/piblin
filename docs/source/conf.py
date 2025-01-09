# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import pathlib

sys.path.insert(0, os.path.abspath('../../src/piblin'))
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

print("*********")

project = 'piblin'
copyright = '2024, 3M'
author = 'Matthew , MPardakhti'
release = '0.0.0a1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    # import the modules you're documenting and pull in documentation from docstrings in a semi-automatic way
    'sphinx.ext.autosummary',
    # generate function/method/attribute summary lists
    'sphinx.ext.napoleon',
    # converts NumPy-style docstrings to reStructuredText prior to autodoc processing
    'sphinx.ext.viewcode'
    # generates HTML page for each module with a highlighted version of the source code - HTML related builders only
]


numpydoc_show_class_member = False
# numpydoc_class_members_toctree = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []