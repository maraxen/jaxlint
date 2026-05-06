"""Griffe-backed docstring parsing (ported from jaxbeans style_validator)."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional, cast

from griffe import Docstring

logger = logging.getLogger(__name__)

_ParseStyle = Literal["google", "numpy", "sphinx", "auto"]


class StyleValidator:
    """Standards-compliant docstring parser using Griffe."""

    def _normalize_spacing(self, text: str) -> str:
        headers = ["Args:", "Returns:", "Parameters", "Returns", "Example:", "Examples:", "Pseudocode", "Hardware Notes"]
        normalized = text
        for h in headers:
            pattern = r"(?<!\n\n)^(\s*" + re.escape(h) + ")"
            normalized = re.sub(pattern, r"\n\1", normalized, flags=re.MULTILINE)
        return normalized

    def validate(self, docstring_text: str, style: str = "numpy") -> dict[str, Any]:
        if not docstring_text:
            return {"passed": False, "sections": [], "error": "Empty docstring"}

        try:
            text = self._normalize_spacing(docstring_text)
            doc = Docstring(text)

            parser_style: str | _ParseStyle = style
            if style in ("optax", "sphinx"):
                parser_style = "sphinx" if ":param" in text else "google"

            sections_parsed = doc.parse(cast(Optional[_ParseStyle], parser_style))

            found_sections: list[str] = []
            for section in sections_parsed:
                kind = section.kind.value if hasattr(section.kind, "value") else str(section.kind)
                found_sections.append(kind)

            has_params = any(s in ("parameters", "args", "arguments") for s in found_sections)
            has_returns = any(s in ("returns", "return") for s in found_sections)
            passed = has_params and has_returns

            return {
                "passed": passed,
                "sections": found_sections,
                "summary": docstring_text.split("\n")[0] if docstring_text else "",
                "parsed_sections": sections_parsed,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("Griffe parsing failed: %s", e)
            return {"passed": False, "sections": [], "error": str(e)}

    def extract_longer_description(self, docstring_text: str, style: str = "numpy") -> str:
        if not docstring_text:
            return ""

        try:
            text = self._normalize_spacing(docstring_text)
            parser_style: str | _ParseStyle = "google" if style in ("sphinx", "optax") else style
            if style == "sphinx" and ":param" in text:
                parser_style = "sphinx"

            doc = Docstring(text)
            sections_parsed = doc.parse(cast(Optional[_ParseStyle], parser_style))

            for section in sections_parsed:
                if section.kind.value == "text":
                    content = section.value
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1:
                        return parts[1].strip()

            return ""
        except Exception:  # noqa: BLE001
            return ""

