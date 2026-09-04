from __future__ import annotations

import re
from typing import Any

REFERENCE_PATTERNS = [
    r"प्र\.?\s*क्र\.?",
    r"जा\.?\s*क्र\.?",
    r"क्रमांक",
    r"\bक्र\.",
    r"\bका\.",
    r"अधिसूचना\s*क्र",
    r"शासन\s*निर्णय",
    r"शासन\s*परिपत्रक",
    r"संदर्भ",
    r"वाचा",
    r"दिनांक|\bदि\.",
]


def reference_pattern_score(text: str) -> float:
    score = 0.0
    for pat in REFERENCE_PATTERNS:
        if re.search(pat, text):
            score += 1.5
    # Slash-separated identifier density
    if re.search(r"[०-९A-Za-zअ-ह]+/[०-९A-Za-zअ-ह\.]+", text):
        score += 2.0
    if text.count("/") >= 2:
        score += 1.0
    return score


def contains_reference_identifier(text: str) -> bool:
    return reference_pattern_score(text) > 0


def detect_reference_features(text: str) -> dict[str, Any]:
    return {
        "reference_pattern_score": reference_pattern_score(text),
        "contains_reference_identifier": contains_reference_identifier(text),
        "slash_count": text.count("/"),
    }
