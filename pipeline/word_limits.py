"""Shared helpers for resolving user-controllable word-length limits."""

from __future__ import annotations

from typing import Any


# Default band: short-to-medium hard crops (user can raise to 50/100+ via CLI or config).
DEFAULT_MIN_WORDS = 2
DEFAULT_MAX_WORDS = 50


def resolve_word_limits(
    cfg: dict[str, Any],
    *,
    min_words: int | None = None,
    max_words: int | None = None,
) -> tuple[int, int]:
    """
    Resolve min/max word counts.

    Priority: explicit CLI args > config.yaml > defaults (2..50).
    """
    lo = DEFAULT_MIN_WORDS if min_words is None else int(min_words)
    hi = DEFAULT_MAX_WORDS if max_words is None else int(max_words)

    if min_words is None:
        lo = int(cfg.get("min_word_count", DEFAULT_MIN_WORDS))
    if max_words is None:
        hi = int(cfg.get("max_word_count", DEFAULT_MAX_WORDS))

    if lo < 1:
        lo = 1
    if hi < lo:
        raise ValueError(f"max_word_count ({hi}) must be >= min_word_count ({lo})")
    return lo, hi


def suggested_max_crop_height(max_words: int, configured: int) -> int:
    """Allow taller crops when the user requests longer text spans."""
    if max_words <= 20:
        return configured
    if max_words <= 50:
        return max(configured, 360)
    if max_words <= 100:
        return max(configured, 520)
    return max(configured, 720)
