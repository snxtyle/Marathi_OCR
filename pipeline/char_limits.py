"""Devanagari-aware character-length limits (primary band for short-token datasets).

Count = Unicode codepoints after NFC, with ZWJ/ZWNJ stripped. Spaces and
punctuation count. Word-length limits remain available; char band is the
primary filter when ``enforce_char_count`` or CLI ``--min-chars`` / ``--max-chars``
is set.
"""

from __future__ import annotations

import unicodedata
from typing import Any

ZWJ = "\u200d"
ZWNJ = "\u200c"

DEFAULT_MIN_CHARS = 4
DEFAULT_MAX_CHARS = 25


def normalize_for_char_count(text: str) -> str:
    cleaned = (text or "").replace(ZWJ, "").replace(ZWNJ, "")
    return unicodedata.normalize("NFC", cleaned)


def char_count(text: str) -> int:
    """Length in NFC codepoints with ZWJ/ZWNJ removed."""
    return len(normalize_for_char_count(text).strip())


def in_char_band(text: str, min_chars: int | None, max_chars: int | None) -> bool:
    n = char_count(text)
    if min_chars is not None and n < min_chars:
        return False
    if max_chars is not None and n > max_chars:
        return False
    return True


def char_band_issues(
    text: str,
    min_chars: int | None,
    max_chars: int | None,
) -> list[str]:
    if min_chars is None and max_chars is None:
        return []
    n = char_count(text)
    issues: list[str] = []
    if min_chars is not None and n < min_chars:
        issues.append(f"too_few_chars:{n}<{min_chars}")
    if max_chars is not None and n > max_chars:
        issues.append(f"too_many_chars:{n}>{max_chars}")
    return issues


def resolve_char_limits(
    cfg: dict[str, Any],
    *,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> tuple[int | None, int | None]:
    """
    Resolve min/max character counts.

    Priority: explicit CLI args > config when enforce_char_count or keys set
    via CLI. If nothing requests a char band, return (None, None) so word
    limits stay the only length filter (factory default).
    """
    enforce = bool(cfg.get("enforce_char_count", False))
    if min_chars is None and max_chars is None and not enforce:
        return None, None

    lo = DEFAULT_MIN_CHARS if min_chars is None else int(min_chars)
    hi = DEFAULT_MAX_CHARS if max_chars is None else int(max_chars)
    if min_chars is None:
        cfg_min = cfg.get("min_char_count")
        lo = int(cfg_min) if cfg_min is not None else DEFAULT_MIN_CHARS
    if max_chars is None:
        cfg_max = cfg.get("max_char_count")
        hi = int(cfg_max) if cfg_max is not None else DEFAULT_MAX_CHARS

    if lo < 1:
        lo = 1
    if hi < lo:
        raise ValueError(f"max_char_count ({hi}) must be >= min_char_count ({lo})")
    return lo, hi


def suggested_min_crop_width(max_chars: int | None, configured: int) -> int:
    """Short character crops are often narrower than full-line 800px min_width."""
    if max_chars is None:
        return configured
    if max_chars <= 25:
        return min(configured, 80)
    if max_chars <= 40:
        return min(configured, 200)
    return configured


# Alias used by extract/validate CLIs
suggested_min_width = suggested_min_crop_width


def suggested_min_aspect(max_chars: int | None, configured: float) -> float:
    if max_chars is None:
        return configured
    if max_chars <= 25:
        return min(configured, 1.15)
    return configured
