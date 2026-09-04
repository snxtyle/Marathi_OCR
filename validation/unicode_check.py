from __future__ import annotations

import unicodedata
from typing import Any

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F


def unicode_issues(text: str) -> list[str]:
    issues: list[str] = []
    if text is None:
        return ["null_text"]
    if "\ufffd" in text:
        issues.append("replacement_character")
    # Strip zero-width chars before deeper checks; they are common in Devanagari PDFs
    cleaned = text.replace("\u200c", "").replace("\u200d", "")
    try:
        cleaned.encode("utf-8")
    except UnicodeEncodeError:
        issues.append("utf8_encode_error")
    nfc = unicodedata.normalize("NFC", cleaned)
    if nfc != cleaned:
        issues.append("not_nfc_normalized")
    for ch in cleaned:
        o = ord(ch)
        if ch.isascii() and ch.isalnum() and ch.isalpha():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in "\n\t":
            issues.append(f"control_char_U+{o:04X}")
    return issues


def is_mostly_devanagari(text: str, min_ratio: float = 0.5) -> bool:
    letters = [c for c in text if c.isalpha() or (DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)]
    if not letters:
        return True
    dev = sum(1 for c in letters if DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)
    return (dev / len(letters)) >= min_ratio


def validate_unicode(text: str) -> dict[str, Any]:
    issues = unicode_issues(text)
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "nfc": unicodedata.normalize("NFC", text or ""),
        "mostly_devanagari": is_mostly_devanagari(text or ""),
    }
