"""jaxlint — JAX-oriented linting and doc quality core (no JAX import in core)."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("jaxlint")
except PackageNotFoundError:
    __version__ = "0.4.2"
