from __future__ import annotations

import re
from typing import Any

from pipeline.char_limits import char_band_issues, char_count
from validation.numerals import validate_marathi_digits_only
from validation.unicode_check import validate_unicode


def ground_truth_checks(
    text: str,
    *,
    marathi_digits_only: bool = True,
    allow_english_text: bool = False,
    min_len: int = 8,
    min_words: int = 2,
    max_words: int = 50,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """
    Automatic sanity checks only.
    MUST NOT claim perfect image-text match — that requires human review.

    Character length (when min_chars/max_chars set) uses NFC codepoint length
    of stripped text — see pipeline.char_limits. When a character band is set,
    it is enforced in addition to word limits (both must pass).
    """
    issues: list[str] = []
    text = text or ""
    if not text.strip():
        issues.append("empty_text")
    floor = min_len if min_chars is None else min_chars
    if char_count(text) < floor and min_chars is None:
        issues.append("too_short")
    if min_chars is not None and char_count(text) < min_chars:
        issues.append("too_short")
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if len(words) < min_words:
        issues.append("too_few_words")
    if len(words) > max_words:
        issues.append(f"too_many_words:{len(words)}>{max_words}")
    issues.extend(char_band_issues(text, min_chars, max_chars))
    if marathi_digits_only:
        ok, msg = validate_marathi_digits_only(text)
        if not ok:
            issues.append(msg)
    if not allow_english_text and re.search(r"[A-Za-z]", text):
        issues.append("latin_letters")
    uni = validate_unicode(text)
    if not uni["ok"]:
        issues.extend(uni["issues"])

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "char_count": char_count(text),
        "image_text_match_verified": False,  # never auto-claim
        "requires_human_review": True,
    }
