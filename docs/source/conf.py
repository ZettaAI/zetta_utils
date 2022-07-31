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
    "sphinx_autodoc_typehints",
    "sphinx.ext.todo",
    "notfound.extension",
    "sphinx_copybutton",
]

doctest_global_setup = """
import zetta_utils as zu
"""

default_role = "any"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "piccolo_theme"
html_show_sphinx = False
html_show_copyright = False
html_static_path = ["_static"]
html_domain_indices = True
html_use_index = True
html_split_index = False
html_show_sourcelink = False

# -- Options for EPUB output
epub_show_urls = "footnote"
