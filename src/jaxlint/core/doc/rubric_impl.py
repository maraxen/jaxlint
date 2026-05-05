"""RubricScorer — multi-layer docstring grading."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from jaxlint.core.doc.style import StyleValidator

logger = logging.getLogger(__name__)

WEIGHT_PRESENCE = 0.10
WEIGHT_STYLE = 0.10
WEIGHT_COMPLETENESS = 0.20
WEIGHT_ACCURACY = 0.20
WEIGHT_CONCISENESS = 0.10
WEIGHT_CLARITY = 0.05
WEIGHT_JAX_IDIOM = 0.20
WEIGHT_FRESHNESS = 0.05

WEIGHTS = {
    "presence": WEIGHT_PRESENCE,
    "style": WEIGHT_STYLE,
    "completeness": WEIGHT_COMPLETENESS,
    "accuracy": WEIGHT_ACCURACY,
    "conciseness": WEIGHT_CONCISENESS,
    "clarity": WEIGHT_CLARITY,
    "jax_idiom": WEIGHT_JAX_IDIOM,
    "freshness": WEIGHT_FRESHNESS,
}


@dataclass
class RubricScore:
    presence: float = 0.0
    style: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    conciseness: float = 0.0
    clarity: float = 0.0
    jax_idiom: float = 0.0
    freshness: float = 0.0
    negative_penalty: float = 0.0
    qs_score: float = 0.0
    qs_scores: dict[str, float] = field(default_factory=dict)
    jax_sharp_bits_multiplier: float = 1.0
    details: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    _score_override: float | None = None
    _grade_override: str | None = None
    _passes_override: bool | None = None

    def set_override(self, score: float, grade: str, passes: bool) -> None:
        self._score_override = score
        self._grade_override = grade
        self._passes_override = passes

    @property
    def raw_score(self) -> float:
        return (
            self.presence
            + self.style
            + self.completeness
            + self.accuracy
            + self.conciseness
            + self.clarity
            + self.jax_idiom
            + self.freshness
            + self.negative_penalty
            + self.qs_score
        )

    @property
    def normalized_score(self) -> float:
        if self._score_override is not None:
            return self._score_override
        total = sum(WEIGHTS.values())
        score = max(0.0, min(1.0, self.raw_score / total))
        return score * self.jax_sharp_bits_multiplier

    @property
    def grade(self) -> str:
        if self._grade_override is not None:
            return self._grade_override
        pct = self.normalized_score * 100
        if pct >= 90:
            return "A+"
        if pct >= 85:
            return "A"
        if pct >= 75:
            return "B"
        if pct >= 60:
            return "C"
        if pct >= 50:
            return "D"
        return "F"

    @property
    def passes(self) -> bool:
        if self._passes_override is not None:
            return self._passes_override
        return self.normalized_score >= 0.75

    def summary(self) -> dict[str, Any]:
        return {
            "raw_score": round(self.raw_score, 2),
            "normalized_score": round(self.normalized_score, 3),
            "grade": self.grade,
            "passes": self.passes,
            "presence": round(self.presence, 2),
            "style": round(self.style, 2),
            "completeness": round(self.completeness, 2),
            "accuracy": round(self.accuracy, 2),
            "conciseness": round(self.conciseness, 2),
            "clarity": round(self.clarity, 2),
            "jax_idiom": round(self.jax_idiom, 2),
            "freshness": round(self.freshness, 2),
            "negative_penalty": round(self.negative_penalty, 2),
            "qs_score": round(self.qs_score, 2),
            "qs_scores": self.qs_scores,
            "jax_sharp_bits_multiplier": round(self.jax_sharp_bits_multiplier, 2),
            "details": self.details,
            "warnings": self.warnings,
        }


class RubricScorer:
    HALLUCINATION_PATTERNS = [
        r"\bDO NOT\b",
        r"\bWARNING\b",
        r"\bTODO\b",
        r"\bDEPRECATED\b",
        r"\bFIXME\b",
        r"\bHACK\b",
    ]

    JAX_PATTERNS = [
        (r"jax\.|jnp\.|jlu\.", "JAX namespace"),
        (r"Array\[", "JAX Array type"),
        (r"Float\[Array|Bool\[Array|Int\[Array", "jaxtyping"),
        (r"PRNGKey|key\(|tree_key", "RNG key handling"),
        (r"pytree|PyTree", "PyTree"),
        (r"vmap|pmap|jit", "JAX transforms"),
    ]

    SHARP_BITS_CRITICAL = [
        (r"\bglobal\s+\w+", "Global state mutation"),
        (r"PRNGKey\(", "Direct PRNGKey creation (use split)"),
    ]

    SHARP_BITS_SUBOPTIMAL = [
        (r"jnp\.log\(jnp\.sum\(jnp\.exp\(", "Missing logsumexp"),
    ]

    def __init__(self, qs_criteria: list[dict] | None = None):
        self.qs_criteria = qs_criteria or []
        self.validator = StyleValidator()

    def score(
        self,
        docstring: str,
        code_signature: str = "",
        code: str = "",
        style: str = "numpy",
    ) -> RubricScore:
        score = RubricScore()
        lines = [line.strip() for line in docstring.split("\n") if line.strip()]

        if not docstring or len(docstring.strip()) < 10:
            score.warnings.append("Docstring missing or too short")
            return score

        self._eval_presence(score, lines)
        self._eval_style(score, docstring, style)
        self._eval_completeness(score, docstring, style)
        self._eval_accuracy(score, docstring)
        self._eval_conciseness(score, docstring)
        self._eval_clarity(score, docstring)
        self._eval_jax_idiom(score, docstring)
        self._eval_freshness(score, docstring, code_signature)
        self._eval_negative_penalty(score, docstring)
        self._eval_jax_sharp_bits(score, docstring, code)
        self._eval_qs_criteria(score, docstring)

        return score

    def _eval_qs_criteria(self, score: RubricScore, docstring: str) -> None:
        for criteria in self.qs_criteria:
            pattern = criteria.get("pattern")
            points = criteria.get("points", 0.0)
            desc = criteria.get("description", "QS match")
            if pattern and re.search(pattern, docstring, re.IGNORECASE):
                score.qs_score += points
                score.qs_scores[desc] = points
                score.details.append(f"+{points}: {desc}")

    def _eval_presence(self, score: RubricScore, lines: list[str]) -> None:
        if lines and len(lines[0]) > 5:
            score.presence = WEIGHT_PRESENCE
            score.details.append(f"+{WEIGHT_PRESENCE}: Docstring exists")

    def _eval_style(self, score: RubricScore, docstring: str, style: str = "numpy") -> None:
        pts = 0.0
        first_line = docstring.split("\n")[0].strip() if docstring else ""

        if first_line.endswith("."):
            pts += WEIGHT_STYLE * 0.2

        val_res = self.validator.validate(docstring, style)
        if val_res.get("passed"):
            pts += WEIGHT_STYLE * 0.8
            score.details.append(f"+{WEIGHT_STYLE}: Style passed ({style})")

        score.style = min(pts, WEIGHT_STYLE)

    def _eval_completeness(self, score: RubricScore, docstring: str, style: str = "numpy") -> None:
        pts = 0.0
        val_res = self.validator.validate(docstring, style)
        sections = val_res.get("sections", [])

        required_names = ["parameters", "returns"]
        if "pseudocode" in docstring.lower():
            pts += WEIGHT_COMPLETENESS * 0.5
            score.details.append("+Pseudocode present")

        found_count = sum(1 for name in required_names if name in sections)
        pts += (found_count / len(required_names)) * (WEIGHT_COMPLETENESS * 0.5)

        score.completeness = min(pts, WEIGHT_COMPLETENESS)
        if pts >= WEIGHT_COMPLETENESS * 0.8:
            score.details.append("+Completeness passed")

    def _eval_accuracy(self, score: RubricScore, docstring: str) -> None:
        pts = 0.0

        if re.search(r"\$\$|\$[^$]+\$", docstring):
            pts += WEIGHT_ACCURACY * 0.6
            score.details.append("+MathJax present")

        math_blocks = re.findall(r"\$\$(.*?)\$\$", docstring, re.DOTALL)
        if math_blocks:
            pts += WEIGHT_ACCURACY * 0.4

        score.accuracy = min(pts, WEIGHT_ACCURACY)

    def _eval_conciseness(self, score: RubricScore, docstring: str) -> None:
        length = len(docstring)
        if 100 <= length <= 600:
            score.conciseness = WEIGHT_CONCISENESS
        elif length < 100:
            score.conciseness = WEIGHT_CONCISENESS * 0.7
        else:
            score.conciseness = WEIGHT_CONCISENESS * 0.5

    def _eval_clarity(self, score: RubricScore, docstring: str) -> None:  # noqa: ARG002
        score.clarity = WEIGHT_CLARITY * 0.5

    def _eval_jax_idiom(self, score: RubricScore, docstring: str) -> None:
        pts = 0.0
        for pattern, name in self.JAX_PATTERNS:
            if re.search(pattern, docstring, re.IGNORECASE):
                pts += (WEIGHT_JAX_IDIOM * 0.6) / len(self.JAX_PATTERNS)
                score.details.append(f"+{name}")

        if "hardware notes" in docstring.lower():
            pts += WEIGHT_JAX_IDIOM * 0.4
            score.details.append("+Hardware Notes present")

        score.jax_idiom = min(pts, WEIGHT_JAX_IDIOM)

    def _eval_freshness(self, score: RubricScore, docstring: str, code_sig: str) -> None:
        if not code_sig:
            score.freshness = WEIGHT_FRESHNESS * 0.5
            return

        doc_sig_match = re.search(r"def\s+(\w+)\s*\(", docstring)
        if doc_sig_match and doc_sig_match.group(1) in code_sig:
            score.freshness = WEIGHT_FRESHNESS

    def _eval_negative_penalty(self, score: RubricScore, docstring: str) -> None:
        penalty = 0.0
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, docstring, re.IGNORECASE):
                penalty -= 0.03

        score.negative_penalty = max(penalty, -0.15)

    def _eval_jax_sharp_bits(self, score: RubricScore, docstring: str, code: str) -> None:  # noqa: ARG002
        target = code
        if not target:
            return

        for pattern, name in self.SHARP_BITS_CRITICAL:
            if re.search(pattern, target):
                score.jax_sharp_bits_multiplier = 0.0
                score.warnings.append(f"CRITICAL: {name}")
                return

        for pattern, name in self.SHARP_BITS_SUBOPTIMAL:
            if re.search(pattern, target):
                score.jax_sharp_bits_multiplier *= 0.5
                score.warnings.append(f"SUBOPTIMAL: {name}")


def quick_score(docstring: str) -> dict[str, Any]:
    return RubricScorer().score(docstring).summary()
