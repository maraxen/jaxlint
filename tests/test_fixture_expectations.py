"""Fixture -> expected-rule-IDs harness (spec 261016, Workstream 1 / J2).

Every ``tests/fixtures/**/*.py`` file declares its expected diagnostics in its
first lines::

    # jaxlint-expect: JL003, JL004      (comma-separated IDs, order irrelevant)
    # jaxlint-expect: none              (no diagnostics at all)
    # jaxlint-config: strict-shapes=true   (optional, second line)

The harness lints each fixture with every rule family enabled (perf, doc,
mathjx) and compares the SET of rule IDs against the header. Multiplicity is
not compared. A fixture without a header fails the run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from jaxlint.core.config import LintConfig
from jaxlint.core.rules import KNOWN_RULE_IDS
from jaxlint.core.runner import diagnostics_for_python_source

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATHS = sorted(FIXTURES_DIR.rglob("*.py"))

_EXPECT_RE = re.compile(r"^#\s*jaxlint-expect:\s*(.*?)\s*$")
_CONFIG_RE = re.compile(r"^#\s*jaxlint-config:\s*(.*?)\s*$")
_RULE_ID_RE = re.compile(r"^(JL|JD|JM)\d{3}$")
_HEADER_LINES = 2
_CONFIG_KEYS = {"strict-shapes"}


def _parse_bool(key: str, value: str) -> bool:
    if value in {"true", "false"}:
        return value == "true"
    msg = f"jaxlint-config {key}= must be true or false, got {value!r}"
    raise ValueError(msg)


def parse_header(path: Path) -> tuple[frozenset[str], LintConfig]:
    """Return ``(expected_rule_ids, lint_config)`` from a fixture's header.

    Raises ``AssertionError`` with a message naming the file when the header is
    missing or malformed, so a bad fixture fails loudly rather than silently
    matching nothing.
    """
    lines = path.read_text(encoding="utf-8").splitlines()[:_HEADER_LINES]
    expect_value: str | None = None
    config_value: str | None = None
    for line in lines:
        if m := _EXPECT_RE.match(line):
            expect_value = m.group(1)
        elif m := _CONFIG_RE.match(line):
            config_value = m.group(1)
    rel = _display(path)
    if expect_value is None:
        msg = (
            f"{rel}: missing '# jaxlint-expect: ...' header in its first "
            f"{_HEADER_LINES} lines (use 'none' for a clean fixture)"
        )
        raise AssertionError(msg)
    if expect_value == "none":
        expected: frozenset[str] = frozenset()
    else:
        ids = [part.strip() for part in expect_value.split(",")]
        bad = [i for i in ids if not _RULE_ID_RE.match(i)]
        if bad or not all(ids):
            msg = f"{rel}: malformed jaxlint-expect value {expect_value!r} (bad: {bad or ids})"
            raise AssertionError(msg)
        expected = frozenset(ids)

    strict_shapes = False
    if config_value is not None:
        for pair in config_value.split():
            key, sep, value = pair.partition("=")
            if not sep or key not in _CONFIG_KEYS:
                msg = f"{rel}: unknown jaxlint-config entry {pair!r} (known keys: {sorted(_CONFIG_KEYS)})"
                raise AssertionError(msg)
            strict_shapes = _parse_bool(key, value)
    return expected, LintConfig(strict_shapes=strict_shapes)


def _display(path: Path) -> str:
    """Path for messages: relative to ``tests/`` when inside it, else the bare name."""
    try:
        return path.relative_to(FIXTURES_DIR.parent).as_posix()
    except ValueError:
        return path.name


def _rel_id(path: Path) -> str:
    return path.relative_to(FIXTURES_DIR).as_posix()


@pytest.mark.parametrize("fixture", FIXTURE_PATHS, ids=[_rel_id(p) for p in FIXTURE_PATHS])
def test_fixture_rule_ids_match_header(fixture: Path) -> None:
    expected, cfg = parse_header(fixture)
    source = fixture.read_text(encoding="utf-8")
    diags = diagnostics_for_python_source(
        fixture,
        source,
        cfg=cfg,
        perf=True,
        doc=True,
        mathjx=True,
    )
    actual = frozenset(d.rule_id for d in diags)
    assert actual == expected, (
        f"{_rel_id(fixture)}: expected {sorted(expected) or 'none'}, "
        f"got {sorted(actual) or 'none'}; "
        f"diagnostics: {[(d.rule_id, d.line) for d in diags]}"
    )


def test_every_fixture_has_header() -> None:
    """Fail with one message listing all fixtures without a header (not just the first)."""
    missing: list[str] = []
    for path in FIXTURE_PATHS:
        try:
            parse_header(path)
        except AssertionError as exc:
            missing.append(str(exc))
    assert not missing, "fixtures without a valid header:\n" + "\n".join(missing)


def test_every_known_rule_has_a_positive_fixture() -> None:
    """Each rule in KNOWN_RULE_IDS must appear in at least one fixture's expect header."""
    covered: set[str] = set()
    for path in FIXTURE_PATHS:
        expected, _ = parse_header(path)
        covered |= expected
    uncovered = sorted(KNOWN_RULE_IDS - covered)
    assert not uncovered, f"rules with no positive fixture: {uncovered}"


def test_parse_header_reads_config_line(tmp_path: Path) -> None:
    f = tmp_path / "cfg.py"
    f.write_text(
        "# jaxlint-expect: JL005\n# jaxlint-config: strict-shapes=true\nx = 1\n", encoding="utf-8"
    )
    expected, cfg = parse_header(f)
    assert expected == frozenset({"JL005"})
    assert cfg.strict_shapes is True


def test_parse_header_rejects_unknown_config_key(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("# jaxlint-expect: none\n# jaxlint-config: strcit-shapes=true\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="unknown jaxlint-config"):
        parse_header(f)
