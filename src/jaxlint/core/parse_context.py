"""Per-file Python source + optional parsed AST (single parse at file boundary)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ParsedPythonFile:
    """Holds path, source text, and ``ast.Module`` after a successful parse.

    On ``OSError`` when reading the file, ``source`` is empty and ``tree`` is
    ``None``. On ``SyntaxError`` after a successful read, ``source`` is the file
    text and ``tree`` is ``None``.
    """

    path: Path
    source: str
    tree: ast.Module | None


def load_parsed_python_file(path: Path) -> ParsedPythonFile:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ParsedPythonFile(path=path, source="", tree=None)
    return parse_python_source(path, source)


def parse_python_source(path: Path, source: str) -> ParsedPythonFile:
    """Parse ``source`` as Python for ``path`` (no disk read).

    On ``SyntaxError``, ``tree`` is ``None`` and ``source`` is preserved (same
    contract as :func:`load_parsed_python_file` after a successful read).
    """

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return ParsedPythonFile(path=path, source=source, tree=None)
    return ParsedPythonFile(path=path, source=source, tree=tree)
