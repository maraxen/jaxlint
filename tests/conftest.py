"""Shared pytest helpers for jaxlint."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def jl001_fixture(fixtures_dir: Path) -> Path:
    return fixtures_dir / "jl001_scatter_gather.py"


@pytest.fixture
def clean_fixture(fixtures_dir: Path) -> Path:
    return fixtures_dir / "clean_minimal.py"
