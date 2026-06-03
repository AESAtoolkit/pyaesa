"""Sphinx configuration for pyaesa documentation."""

import re
import sys
from pathlib import Path

from docutils import nodes
from sphinx.application import Sphinx

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
html_js_files = ["custom.js"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}

PYAESA_BRAND_PATTERN = re.compile(r"(?<![A-Za-z0-9_./:=@-])pyaesa(?![A-Za-z0-9_./:-])")
PYAESA_IMPORT_PATTERN = re.compile(r"\b(from\s+pyaesa\s+import|import\s+pyaesa)\b")
PYAESA_SKIP_TEXT_ANCESTORS = (
    nodes.literal,
    nodes.literal_block,
    nodes.raw,
    nodes.math,
    nodes.math_block,
    nodes.problematic,
)
PYAESA_SKIP_LITERAL_ANCESTORS = (
    nodes.literal_block,
    nodes.raw,
    nodes.math,
    nodes.math_block,
    nodes.problematic,
)


def _brand_inline(text: str, css_class: str) -> nodes.inline:
    """Return one styled inline node for the package brand."""
    return nodes.inline(text, text, classes=[css_class])


def _brand_node() -> nodes.inline:
    """Return the unbreakable styled pyaesa brand node."""
    brand = nodes.inline("", "", classes=["pyaesa-brand"])
    brand += _brand_inline("py", "pyaesa-brand-py")
    brand += _brand_inline("aesa", "pyaesa-brand-aesa")
    return brand


def _node_has_skip_ancestor(node: nodes.Node, skip_ancestors: tuple[type[nodes.Node], ...]) -> bool:
    """Return whether the node is nested in an element excluded from branding."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, skip_ancestors):
            return True
        parent = parent.parent
    return False


def _text_node_allows_branding(text_node: nodes.Text) -> bool:
    """Return whether a text node is visible prose eligible for brand styling."""
    text = text_node.astext()
    if not text.strip():
        return False
    if PYAESA_IMPORT_PATTERN.search(text):
        return False
    return not _node_has_skip_ancestor(text_node, PYAESA_SKIP_TEXT_ANCESTORS)


def _literal_node_allows_branding(literal_node: nodes.literal) -> bool:
    """Return whether an exact pyaesa inline literal should become brand spans."""
    if literal_node.astext() != "pyaesa":
        return False
    return not _node_has_skip_ancestor(literal_node, PYAESA_SKIP_LITERAL_ANCESTORS)


def _brand_replacement_nodes(text: str) -> list[nodes.Node]:
    """Split text into docutils nodes with styled pyaesa brand spans."""
    replacements: list[nodes.Node] = []
    position = 0
    for match in PYAESA_BRAND_PATTERN.finditer(text):
        if match.start() > position:
            replacements.append(nodes.Text(text[position : match.start()]))
        replacements.append(_brand_node())
        position = match.end()
    if position < len(text):
        replacements.append(nodes.Text(text[position:]))
    return replacements


def _apply_pyaesa_branding(doctree: nodes.document) -> None:
    """Apply the Read the Docs pyaesa brand styling to visible prose nodes."""
    for literal_node in list(doctree.findall(nodes.literal)):
        if not _literal_node_allows_branding(literal_node):
            continue
        if literal_node.parent is None:
            continue
        literal_node.parent.replace(literal_node, _brand_replacement_nodes("pyaesa"))

    for text_node in list(doctree.findall(nodes.Text)):
        if not _text_node_allows_branding(text_node):
            continue
        text = text_node.astext()
        if not PYAESA_BRAND_PATTERN.search(text):
            continue
        if text_node.parent is None:
            continue
        text_node.parent.replace(text_node, _brand_replacement_nodes(text))


def apply_pyaesa_branding_on_read(app: Sphinx, doctree: nodes.document) -> None:
    """Apply pyaesa brand styling before Sphinx stores document titles."""
    del app
    _apply_pyaesa_branding(doctree)


def apply_pyaesa_branding_on_resolved(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Apply pyaesa brand styling after Sphinx resolves cross document content."""
    del app, docname
    _apply_pyaesa_branding(doctree)


def copy_methodological_notes(app: Sphinx, exception: Exception | None) -> None:
    """Copy repository methodological PDFs into the HTML output tree."""
    if exception is not None:
        return
    source = ROOT / "methodological_notes"
    destination = Path(app.outdir) / "methodological_notes"
    if source.exists():
        from shutil import copytree

        copytree(source, destination, dirs_exist_ok=True)


def setup(app: Sphinx) -> None:
    app.connect("doctree-read", apply_pyaesa_branding_on_read)
    app.connect("doctree-resolved", apply_pyaesa_branding_on_resolved)
    app.connect("build-finished", copy_methodological_notes)
