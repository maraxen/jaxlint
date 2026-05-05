# jaxlint Sphinx configuration (Read the Docs + local builds).
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
from datetime import UTC, datetime


def _canonical_github_https() -> str:
    """Prefer RTD-provided clone URL when available."""
    clone = os.environ.get("READTHEDOCS_GIT_CLONE_URL", "")
    if clone.startswith("git@github.com:"):
        path = clone.removeprefix("git@github.com:").removesuffix(".git")
        return f"https://github.com/{path}"
    if clone.startswith("https://github.com/"):
        return clone.removesuffix(".git").removesuffix("/")
    return "https://github.com/jax-ml/jaxlint"


_repo_url = _canonical_github_https()
_project = "jaxlint"
_year = datetime.now(UTC).year

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "myst",
    ".rst": "restructuredtext",
}

myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

# Read the Docs + jax-ml org URLs: ignored for linkcheck until those endpoints are
# reliably reachable (404/rate-limit otherwise). Prefer tightening this list over time.
linkcheck_ignore = [
    r"^https://jaxlint\.readthedocs\.io",
    r"^https://github\.com/jax-ml/jaxlint",
]
linkcheck_timeout = 15
linkcheck_retries = 2

html_theme = "sphinx_book_theme"
html_title = f"{_project} documentation"
html_theme_options = {
    "repository_url": _repo_url,
    "use_repository_button": True,
    "use_issues_button": False,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
html_static_path: list[str] = []

html_context = {
    "display_github": True,
    "github_url": _repo_url,
}

if os.environ.get("READTHEDOCS_PROJECT"):
    canonical = os.environ.get("READTHEDOCS_CANONICAL_URL", "").rstrip("/")
    if canonical:
        html_baseurl = f"{canonical}/"

project = _project
copyright = f"{_year}, jaxlint authors"
author = "jaxlint authors"
version = os.environ.get("READTHEDOCS_VERSION", "") or ""
release = version

pygments_style = "friendly"
copybutton_prompt_text = r"$ "

# Better default for typography (em-dash, typography quotes) without extra TeX tweaking.
latex_engine = "xelatex"
