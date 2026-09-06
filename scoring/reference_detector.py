"""Generic identifier-likeness signals (no language/keyword lists).

Hard/complex lane decisions must NOT use Maharashtra-GR templates
(जिपास, प्र.क्र., मभाप-, etc.). This module only measures structural
digit+separator density that is useful as a soft ranking feature.
"""

from __future__ import annotations

import re
from typing import Any

# Digit run adjacent to separators — script-agnostic structure, not GR keywords
_DIGIT_SEP_RE = re.compile(
    r"[०-९0-9]{1,4}\s*[./\-–—:]\s*[०-९0-9A-Za-zअ-ह]{1,8}"
)
_MULTI_SEP_DIGIT_RE = re.compile(
    r"[०-९0-9]+(?:\s*[./\-–—]\s*[०-९0-9A-Za-zअ-ह]+){2,}"
)


def identifier_likeness_score(text: str) -> float:
    """Soft structural score for ID-like digit/separator clusters.

    Deliberately has no keyword list. A single slash in prose scores 0.
    """
    t = text or ""
    score = 0.0
    if _MULTI_SEP_DIGIT_RE.search(t):
        score += 2.0
    elif _DIGIT_SEP_RE.search(t):
        score += 1.0
    digit_n = sum(1 for c in t if c in "०१२३४५६७८९0123456789")
    seps = t.count("/") + t.count("-") + t.count(".") + t.count(":")
    if digit_n >= 3 and seps >= 2:
        score += 0.75
    return score


# Back-compat aliases — former keyword-list hardness path is gone
def reference_pattern_score(text: str) -> float:
    return identifier_likeness_score(text)


def contains_reference_identifier(text: str) -> bool:
    return identifier_likeness_score(text) > 0


def detect_reference_features(text: str) -> dict[str, Any]:
    return {
        "identifier_likeness_score": identifier_likeness_score(text),
        "reference_pattern_score": identifier_likeness_score(text),
        "contains_reference_identifier": contains_reference_identifier(text),
        "slash_count": (text or "").count("/"),
    }
