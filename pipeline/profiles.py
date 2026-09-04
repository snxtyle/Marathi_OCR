"""Active dataset profile helpers.

Numeral / ASCII policy lives on the profile, not the global OCR factory.
"""

from __future__ import annotations

from typing import Any


def get_active_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    name = str(cfg.get("active_profile") or "marathi_hard_validation")
    profiles = cfg.get("profiles") or {}
    profile = dict(profiles.get(name) or {})
    profile["_name"] = name
    # Fallbacks from top-level config for older callers
    if "allow_ascii_digits" not in profile:
        profile["allow_ascii_digits"] = not bool(cfg.get("marathi_digits_only", True))
    if "marathi_digits_only" not in profile:
        profile["marathi_digits_only"] = not bool(profile.get("allow_ascii_digits", False))
    if "hard_count" not in profile:
        profile["hard_count"] = int(cfg.get("validation_hard_count", 80))
    if "normal_count" not in profile:
        profile["normal_count"] = int(cfg.get("validation_normal_count", 20))
    if "max_samples_per_source" not in profile:
        profile["max_samples_per_source"] = 5
    return profile


def allow_ascii_digits(cfg: dict[str, Any]) -> bool:
    return bool(get_active_profile(cfg).get("allow_ascii_digits", False))


def marathi_digits_only(cfg: dict[str, Any]) -> bool:
    return not allow_ascii_digits(cfg)
