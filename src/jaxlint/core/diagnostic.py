"""Unified diagnostic model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    rule_id: str
    severity: Severity
    message: str
    path: Path
    line: int
    col: int = 0
    end_line: int | None = None
    end_col: int | None = None

    def sort_key(self) -> tuple:
        return (str(self.path), self.line, self.col, self.rule_id)
