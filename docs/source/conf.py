# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "zetta_utils"
copyright = "2022, Zetta AI"
author = "Sergiy Popovych"

release = "0.0"
version = "0.0.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    #    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "piccolo_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
