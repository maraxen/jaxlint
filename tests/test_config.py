from __future__ import annotations

from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig, effective_severity, load_config, rule_enabled
from jaxlint.core.diagnostic import Diagnostic, Severity


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
def test_effective_severity_override_wins_over_emitted(tmp_path: Path) -> None:
    p = tmp_path / "f.py"
    d = Diagnostic("JL005", Severity.ERROR, "shape", p, line=1, col=0)
    cfg = LintConfig(severity_overrides={"JL005": Severity.WARNING})
    assert effective_severity(d, cfg).severity == Severity.WARNING


@pytest.mark.unit
def test_load_config_jaxlint_toml_over_pyproject_in_walk(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.jaxlint]\ndocstring-style = "google"\n',
        encoding="utf-8",
    )
    (pkg / "jaxlint.toml").write_text(
        '[jaxlint]\ndocstring-style = "numpy"\n',
        encoding="utf-8",
    )
    cfg = load_config(pkg)
    assert cfg.docstring_style == "numpy"


@pytest.mark.unit
def test_load_config_conflict_same_directory_raises(tmp_path: Path) -> None:
    d = tmp_path / "here"
    d.mkdir()
    (d / "jaxlint.toml").write_text("[tool.jaxlint]\n", encoding="utf-8")
    (d / "pyproject.toml").write_text("[tool.jaxlint]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Both jaxlint.toml and pyproject.toml"):
        load_config(d)


@pytest.mark.unit
def test_load_config_direct_config_file(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pyproject.toml"
    cfg_path.write_text('[tool.jaxlint]\nstrict-shapes = true\n', encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.strict_shapes is True


@pytest.mark.unit
def test_load_config_direct_jaxlint_file(tmp_path: Path) -> None:
    jxl = tmp_path / "jaxlint.toml"
    jxl.write_text('[tool.jaxlint]\nstrict-shapes = true\n', encoding="utf-8")
    assert load_config(jxl).strict_shapes is True


@pytest.mark.unit
def test_load_config_file_must_be_named(tmp_path: Path) -> None:
    other = tmp_path / "other.toml"
    other.write_text("[tool.jaxlint]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jaxlint.toml"):
        load_config(other)


@pytest.mark.unit
def test_severity_overrides_invalid_severity_raises(tmp_path: Path) -> None:
    p = tmp_path / "pyproject.toml"
    p.write_text(
        '[tool.jaxlint]\nseverity-overrides = { JL005 = "oops" }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid severity"):
        load_config(p)


@pytest.mark.unit
def test_severity_overrides_unknown_rule_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "pyproject.toml"
    p.write_text(
        '[tool.jaxlint]\nseverity-overrides = { XX999 = "warning" }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match any known rule"):
        load_config(p)


@pytest.mark.unit
def test_severity_overrides_nested_table(tmp_path: Path) -> None:
    p = tmp_path / "pyproject.toml"
    p.write_text(
        '[tool.jaxlint]\nstrict-shapes = true\n\n[tool.jaxlint.severity-overrides]\nJL005 = "warning"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.strict_shapes is True
    assert cfg.severity_overrides["JL005"] == Severity.WARNING


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
def test_load_config_jaxlint_toml_native_section_only(tmp_path: Path) -> None:
    jxl = tmp_path / "jaxlint.toml"
    jxl.write_text(
        """
[jaxlint]
select = ["JL"]
strict-shapes = true
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(jxl)
    assert cfg.select == frozenset(["JL"])
    assert cfg.strict_shapes is True


@pytest.mark.unit
def test_load_config_jaxlint_toml_both_sections_raises(tmp_path: Path) -> None:
    jxl = tmp_path / "jaxlint.toml"
    jxl.write_text(
        """
[jaxlint]
strict-shapes = true

[tool.jaxlint]
docstring-style = "google"
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both \\[jaxlint\\] and \\[tool\\.jaxlint\\]"):
        load_config(jxl)


@pytest.mark.unit
def test_load_config_pyproject_root_jaxlint_section_raises(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text(
        """
[jaxlint]
strict-shapes = true

[tool.jaxlint]
docstring-style = "numpy"
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="\\[tool\\.jaxlint\\]"):
        load_config(pp)


@pytest.mark.unit
def test_native_and_legacy_jaxlint_toml_equivalent(tmp_path: Path) -> None:
    native = tmp_path / "native" / "jaxlint.toml"
    legacy = tmp_path / "legacy" / "jaxlint.toml"
    native.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    native.write_text(
        """
[jaxlint]
select = ["JL", "JM"]
ignore = ["JM001"]
docstring-style = "numpy"
required-sections = ["Parameters", "Returns"]
section-order = ["Parameters", "Returns", "Raises"]
strict-shapes = true
rubric-threshold = 0.5

[jaxlint.per-file-ignores]
"**/legacy_*.py" = "JL012"
"a/b.py" = ["JD001"]

[jaxlint.severity-overrides]
JL005 = "warning"
""".strip(),
        encoding="utf-8",
    )
    legacy.write_text(
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

[tool.jaxlint.severity-overrides]
JL005 = "warning"
""".strip(),
        encoding="utf-8",
    )
    assert load_config(native) == load_config(legacy)


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
