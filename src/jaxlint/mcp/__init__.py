"""Optional Model Context Protocol (MCP) integration for jaxlint."""

from __future__ import annotations

from jaxlint.mcp.server import (
    jaxlint_check_impl,
    jaxlint_rules_text,
    jaxlint_version_text,
    main,
    mcp,
)

__all__ = ["jaxlint_check_impl", "jaxlint_rules_text", "jaxlint_version_text", "main", "mcp"]
