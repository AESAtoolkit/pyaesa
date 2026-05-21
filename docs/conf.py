"""Sphinx configuration for pyaesa documentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

project = "pyaesa"
author = "pyaesa contributors"

try:
    import pyaesa

    release = getattr(pyaesa, "__version__", "0.0.0")
except Exception:
    release = "0.0.0"

extensions: list[str] = [
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
myst_enable_extensions: list[str] = ["colon_fence"]
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
suppress_warnings = ["config.cache"]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_logo = "../images/fig-pyaesa-logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
