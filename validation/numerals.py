from __future__ import annotations

import re

ASCII_DIGIT_RE = re.compile(r"[0-9]")
MARATHI_DIGIT_RE = re.compile(r"[०-९]")


def has_ascii_digits(text: str) -> bool:
    return bool(ASCII_DIGIT_RE.search(text or ""))


# Alias used by extraction / profile gates
contains_ascii_digits = has_ascii_digits


def has_marathi_digits(text: str) -> bool:
    return bool(MARATHI_DIGIT_RE.search(text or ""))


def validate_marathi_digits_only(text: str) -> tuple[bool, str]:
    """Reject expected_text containing Arabic/ASCII digits 0-9."""
    if has_ascii_digits(text):
        return False, "expected_text contains ASCII digits 0-9; only ०-९ allowed"
    return True, ""


def ascii_to_marathi_digits(text: str) -> str:
    return text.translate(str.maketrans("0123456789", "०१२३४५६७८९"))
