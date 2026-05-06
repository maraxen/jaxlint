"""Shared pytest helpers for jaxlint."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Bound wall-clock time for JAX/HLO tests (see docs/hlo.md — goldens)."""
    mark = pytest.mark.timeout(60)
    for item in items:
        if item.get_closest_marker("needs_jax"):
            item.add_marker(mark)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def jl001_fixture(fixtures_dir: Path) -> Path:
    return fixtures_dir / "jl001_scatter_gather.py"


@pytest.fixture
def clean_fixture(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clean_minimal.py"
