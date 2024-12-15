# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the directories containing your modules to the system path
sys.path.insert(0, os.path.abspath('../'))  # Adjust the path based on your project structure

# Now you can import your modules
sys.path.insert(0, os.path.abspath(os.path.join('..', 'api_extract_prep')))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'modelling')))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TFM_IDEALISTA_API_PRICE_PRED'
copyright = '2024, Verónica García de Olalla Blancafort'
author = 'Verónica García de Olalla Blancafort'
release = '15/12/2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
