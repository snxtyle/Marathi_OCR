"""Lane quota helpers — scale hard/normal from profile ratio to a target count."""

from __future__ import annotations

from typing import Any


def base_lane_counts(profile: dict[str, Any], cfg: dict[str, Any] | None = None) -> tuple[int, int]:
    """Return (hard, normal) base counts from profile / config defaults."""
    cfg = cfg or {}
    hard = int(profile.get("hard_count") or cfg.get("validation_hard_count", 80))
    normal = int(profile.get("normal_count") or cfg.get("validation_normal_count", 20))
    return max(0, hard), max(0, normal)


def scale_lane_targets(
    *,
    target_count: int | None = None,
    hard_count: int | None = None,
    normal_count: int | None = None,
    profile: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Resolve hard/normal targets.

    Priority:
      1. Explicit ``hard_count`` + ``normal_count``
      2. ``target_count`` scaled by profile ratio (default 80/20)
      3. Profile / config base counts

    When only one of hard/normal is explicit with ``target_count``, the other
    fills the remainder. Sum always equals ``target_count`` when that is set
    and neither lane override is used alone incorrectly.
    """
    profile = profile or {}
    cfg = cfg or {}
    base_h, base_n = base_lane_counts(profile, cfg)
    base_total = max(base_h + base_n, 1)

    if hard_count is not None and normal_count is not None:
        return max(0, int(hard_count)), max(0, int(normal_count))

    if target_count is not None:
        n = max(0, int(target_count))
        if hard_count is not None:
            h = max(0, min(int(hard_count), n))
            return h, n - h
        if normal_count is not None:
            normal = max(0, min(int(normal_count), n))
            return n - normal, normal
        # Preserve ratio; assign remainder to normal so sum == n
        hard = int(round(n * (base_h / base_total)))
        hard = max(0, min(n, hard))
        # Avoid emptying a lane when base ratio had both > 0 and n is large enough
        if base_h > 0 and base_n > 0 and n >= 2:
            hard = max(1, min(n - 1, hard))
        return hard, n - hard

    if hard_count is not None:
        return max(0, int(hard_count)), base_n
    if normal_count is not None:
        return base_h, max(0, int(normal_count))
    return base_h, base_n


def extract_headroom(hard_target: int, normal_target: int, *, min_top_k: int = 200) -> int:
    """Suggested extract top-k: enough surplus for QA attrition without lowering quality."""
    total = hard_target + normal_target
    # ~4× inventory before OCR/vision attrition; floor at min_top_k
    return max(min_top_k, total * 4)
