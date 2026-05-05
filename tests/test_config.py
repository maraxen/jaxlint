from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig, load_config, rule_enabled


@pytest.fixture
def nested_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    project_root = tmp_path / "proj"
    sub = project_root / "pkg" / "deep"
    sub.mkdir(parents=True)
    toml = project_root / "pyproject.toml"
    toml.write_text(
        """
[tool.jaxlint]
select = ["JL", "JM"]
ignore = ["JM001"]
docstring-style = "numpy"
required-sections = ["Parameters", "Returns"]
section-order = ["Parameters", "Returns", "Raises"]
strict-shapes = true
rubric-threshold = 0.5

[tool.jaxlint.per-file-ignores]
"**/legacy_*.py" = "JL012"
"a/b.py" = ["JD001"]
""".strip(),
        encoding="utf-8",
    )
    return sub


@pytest.mark.unit
def test_load_config_from_nested_walks_parents(nested_pyproject: Path) -> None:
    cfg = load_config(nested_pyproject)
    assert cfg.select == frozenset(["JL", "JM"])
    assert cfg.ignore == frozenset(["JM001"])
    assert cfg.docstring_style == "numpy"
    assert cfg.required_sections == ("Parameters", "Returns")
    assert cfg.section_order == ("Parameters", "Returns", "Raises")
    assert cfg.strict_shapes is True
    assert cfg.rubric_threshold == 0.5
    assert cfg.per_file_ignores["**/legacy_*.py"] == ("JL012",)
    assert cfg.per_file_ignores["a/b.py"] == ("JD001",)


@pytest.mark.unit
def test_load_config_missing_file_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg == LintConfig.default()


@pytest.mark.unit
def test_rule_enabled_select_and_ignore() -> None:
    cfg_all = LintConfig()
    cfg_select = LintConfig(select=frozenset(["JL"]))
    cfg_ignore = LintConfig(select=frozenset(["JL", "JM"]), ignore=frozenset(["JL00"]))

    p = Path("src/model.py")
    assert rule_enabled("JL001", cfg_all, p) is True
    assert rule_enabled("JD001", cfg_select, p) is False
    assert rule_enabled("JL001", cfg_select, p) is True
    assert rule_enabled("JL001", cfg_ignore, p) is False
    assert rule_enabled("JM001", cfg_ignore, p) is True


@pytest.mark.unit
def test_rule_enabled_per_file_string_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = LintConfig(
        select=frozenset(["JL", "JD"]),
        ignore=frozenset(),
        per_file_ignores={
            "**/legacy_*.py": ("JL*",),
            "**/mod.py": ("JD004", "JD005"),
        },
    )
    assert rule_enabled("JL001", cfg, Path("pkg/legacy_foo.py")) is False
    assert rule_enabled("JD004", cfg, Path("x/mod.py")) is False
    assert rule_enabled("JD003", cfg, Path("pkg/legacy_foo.py")) is True
