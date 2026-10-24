import datetime as dt
import importlib.metadata as im
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # src-layout project structure

# Project information ---------------------------------------------------------
project = "Explicit Priors"
author = "Youssef Haouchat"
copyright = f"{dt.date.today().year}, {author}"
try:
    version = im.version("explicit_priors")
except im.PackageNotFoundError:
    version = "0.1.0"
release = version
# -----------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "make_figures.py"]

# Theme options ---------------------------------------------------------------
html_theme = "bizstyle"
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "searchbox.html",
    ]
}

# Extension options -----------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

mathjax3_config = dict(
    tex=dict(
        macros={
            "bbx": r"{\bf{x}}",
            "bby": r"{\bf{y}}",
            "bbm": r"{\bf{m}}",
            "bbf": r"{\bf{f}}",
            "bbc": r"{\bf{c}}",
            "bbd": r"{\bf{d}}",
            "bbA": r"{\bf{A}}",
            "bbG": r"{\bf{G}}",
            "bbW": r"{\bf{W}}",
            "bE": r"{\mathbb{E}}",
            "bR": r"{\mathbb{R}}",
            "argmin": r"\operatorname*{arg\,min}",
            "norm": [r"\left\| #1 \right\|", 1],
        },
    ),
)

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

bibtex_bibfiles = ["bibliography.bib"]
bibtex_reference_style = "label"
