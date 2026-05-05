"""Parity-style test: jaxlint JL001 agrees with inlined legacy scatter+gather heuristic."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.sensors.scatter_gather import scatter_gather_diagnostics


@pytest.mark.unit
def test_scatter_gather_on_fixture(fixtures_dir: Path) -> None:
    root = fixtures_dir / "jl001_scatter_gather.py"
    text = root.read_text(encoding="utf-8")
    sg = scatter_gather_diagnostics(text, root)
    assert any(d.rule_id == "JL001" for d in sg)
