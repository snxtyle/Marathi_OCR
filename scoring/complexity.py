from __future__ import annotations

import re
import unicodedata
from typing import Any

from pipeline.char_limits import char_count, in_char_band
from scoring.reference_detector import identifier_likeness_score

MARATHI_DIGITS = "०१२३४५६७८९"
ASCII_DIGITS = "0123456789"
SPECIAL_CHARS = "/.,-—–:;()[]*%₹«»“”|"
# Devanagari matras / vowel signs
MATRA_RE = re.compile(r"[\u093A-\u094C\u094E\u094F\u0955-\u0957\u0962\u0963]")
VIRAMA = "\u094d"
# Instant-fail hard symbols for ordinary-prose gate.
# Colons/parens/brackets are gated separately via hardish counts so light
# list markers like "(क)" still count as ordinary Marathi.
HARD_PUNCT_RE = re.compile(r"[/₹%—–\*\[\]]")

TRIVIAL_REJECT = [
    re.compile(r"^महाराष्ट्र\.?$"),
    re.compile(r"^आज\s+हवामान"),
    re.compile(r"^अचूक\s+प्रतिमेतील"),
    re.compile(r"^महाराष्ट्र\s+सुंदर"),
]


def _count_chars(text: str, alphabet: str) -> int:
    return sum(text.count(c) for c in alphabet)


def estimate_conjuncts(text: str) -> int:
    return text.count(VIRAMA)


def matra_score(text: str) -> float:
    n = len(MATRA_RE.findall(text or ""))
    return float(min(n, 20)) * 0.08


def compute_features(text: str) -> dict[str, Any]:
    text = unicodedata.normalize("NFC", text or "")
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    length = len(text)
    special = _count_chars(text, SPECIAL_CHARS)
    marathi_digits = _count_chars(text, MARATHI_DIGITS)
    ascii_digits = _count_chars(text, ASCII_DIGITS)
    id_like = identifier_likeness_score(text)
    return {
        "marathi_digit_count": marathi_digits,
        "ascii_digit_count": ascii_digits,
        "slash_count": text.count("/"),
        "period_count": text.count("."),
        "comma_count": text.count(","),
        "hyphen_count": text.count("-"),
        "dash_count": text.count("—") + text.count("–"),
        "colon_count": text.count(":"),
        "semicolon_count": text.count(";"),
        "bracket_count": sum(text.count(c) for c in "()[]{}"),
        "symbol_count": special,
        "currency_count": text.count("₹") + text.count("रु."),
        "percent_count": text.count("%"),
        "conjunct_estimate": estimate_conjuncts(text),
        "matra_score": matra_score(text),
        "text_length": length,
        "char_count": char_count(text),
        "word_count": len(words),
        "numeric_density": (marathi_digits + ascii_digits) / max(length, 1),
        "special_character_density": special / max(length, 1),
        "identifier_likeness_score": id_like,
        # Legacy key kept for older callers; same as identifier_likeness
        "reference_pattern_score": id_like,
        "dense_text_score": min(len(words) / 8.0, 3.0) + min(length / 80.0, 2.0),
    }


def complexity_score(text: str) -> float:
    """Generic OCR-hardness ranking score (no keyword boosts)."""
    f = compute_features(text)
    score = 0.0
    score += min(f["marathi_digit_count"], 12) * 0.45
    score += min(f["slash_count"], 6) * 0.35  # mild; alone does not make hard
    score += min(f["period_count"], 10) * 0.15
    score += min(f["hyphen_count"] + f["dash_count"], 6) * 0.25
    score += min(f["comma_count"] + f["colon_count"], 8) * 0.12
    score += min(f["bracket_count"], 6) * 0.2
    score += min(f["currency_count"] + f["percent_count"], 4) * 0.5
    score += min(f["conjunct_estimate"], 15) * 0.15
    score += f["matra_score"]
    score += f["identifier_likeness_score"] * 0.8
    score += f["dense_text_score"]
    score += f["special_character_density"] * 8
    score += f["numeric_density"] * 8
    score -= f["ascii_digit_count"] * 0.5
    return round(score, 4)


def visual_complexity_score(
    *,
    width: int = 0,
    height: int = 0,
    aspect_ratio: float = 0.0,
    cheap_ocr_text: str = "",
) -> float:
    """Lane-assignment score when PDF text layer is missing (scanned docs)."""
    score = 0.0
    if aspect_ratio >= 4.0:
        score += 1.5
    if width >= 1200:
        score += 0.8
    if 40 <= height <= 180:
        score += 1.0
    if cheap_ocr_text.strip():
        score += complexity_score(cheap_ocr_text) * 0.85
    else:
        # Unknown content — mild prior for line-like crops
        score += 1.2
    return round(score, 4)


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])


def is_too_long(
    text: str,
    max_words: int = 50,
    *,
    max_chars: int | None = None,
) -> bool:
    if word_count(text) > max_words:
        return True
    if max_chars is not None and char_count(text) > max_chars:
        return True
    return False


def is_trivial(
    text: str,
    *,
    min_words: int = 2,
    max_words: int = 50,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> bool:
    t = (text or "").strip()
    n_chars = char_count(t)
    floor = 8 if min_chars is None else min_chars
    if n_chars < floor:
        return True
    if not in_char_band(t, min_chars, max_chars):
        return True
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < min_words:
        return True
    if len(words) > max_words:
        return True
    if len(words) <= 1 and complexity_score(t) < 2.5:
        return True
    for pat in TRIVIAL_REJECT:
        if pat.search(t):
            return True
    return False


def is_ordinary_prose(text: str) -> bool:
    """True if text is ordinary Marathi — suitable for the normal lane."""
    t = unicodedata.normalize("NFC", (text or "").strip())
    if not t:
        return False
    f = compute_features(t)
    if f["marathi_digit_count"] or f["ascii_digit_count"]:
        return False
    if f["currency_count"] or f["percent_count"]:
        return False
    if f["identifier_likeness_score"] > 0:
        return False
    if f["slash_count"] > 0:
        return False
    if HARD_PUNCT_RE.search(t):
        return False
    hardish = f["colon_count"] + f["semicolon_count"] + f["bracket_count"]
    hardish += f["hyphen_count"] + f["dash_count"]
    if hardish > 2:
        return False
    if f["special_character_density"] > 0.06:
        return False
    return True


def strong_generic_hard_signals(features: dict[str, Any]) -> bool:
    """Multi-signal OCR hardness — never keyword-based, never slash-alone.

    Requires numerals, identifier-like digit+separator structure, and/or
    currency/percent. Multi-slash office titles without digits are NOT hard
    (LLM may later promote them if vision judges them complex).
    """
    digits = int(features.get("marathi_digit_count") or 0)
    id_like = float(features.get("identifier_likeness_score") or 0)
    currency = int(features.get("currency_count") or 0) + int(features.get("percent_count") or 0)
    symbols = int(features.get("symbol_count") or 0)
    slash = int(features.get("slash_count") or 0)

    if digits >= 2:
        return True
    if digits >= 1 and (id_like > 0 or slash >= 1 or symbols >= 3):
        return True
    if id_like >= 1.5:
        return True
    if currency >= 1 and (digits >= 1 or symbols >= 2):
        return True
    return False


def classify_difficulty(
    text: str,
    *,
    min_hard_score: float = 3.0,
    max_normal_score: float = 2.5,
    min_words: int = 2,
    max_words: int = 50,
    normal_min_words: int = 6,
    normal_max_words: int = 40,
    min_chars: int | None = None,
    max_chars: int | None = None,
    allow_ascii_digits: bool = False,
    score_override: float | None = None,
    llm_complexity: str | None = None,
) -> dict[str, Any]:
    """Assign hard | normal | reject.

    Heuristic path uses only generic structural signals (digits, punctuation
    density, conjuncts, identifier-likeness). Keyword/GR lists are not used.

    When ``llm_complexity`` is ``hard|normal|reject``, it is authoritative
    (after trivial/ASCII gates). Soft-fail: omit llm_complexity and weak
    slash-only lines will NOT become hard.
    """
    scored = score_candidate(
        text,
        min_words=min_words,
        max_words=max_words,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    if score_override is not None:
        scored["complexity_score"] = score_override
    f = scored["features"]
    reasons: list[str] = []

    if not allow_ascii_digits and f["ascii_digit_count"] > 0:
        return {
            **scored,
            "difficulty": "reject",
            "reject_reasons": ["ascii_digits"],
            "hard_signal_source": "none",
        }

    llm_label = (llm_complexity or "").strip().lower()
    if llm_label in {"hard", "normal", "reject"}:
        if llm_label == "reject":
            return {
                **scored,
                "difficulty": "reject",
                "reject_reasons": ["llm_complexity_reject"],
                "hard_signal_source": "llm",
            }
        if scored["is_trivial"] or scored.get("is_too_long"):
            if not (
                f.get("marathi_digit_count", 0) > 0
                or f.get("identifier_likeness_score", 0) > 0
            ):
                return {
                    **scored,
                    "difficulty": "reject",
                    "reject_reasons": ["trivial_or_too_long"],
                    "hard_signal_source": "llm",
                }
        wc_llm = int(scored["word_count"])
        # Never let LLM force ordinary flowing prose into the hard lane.
        # LLM may still promote weak slash/title lines (not ordinary) to hard.
        if (
            llm_label == "hard"
            and is_ordinary_prose(text)
            and normal_min_words <= wc_llm <= normal_max_words
            and not strong_generic_hard_signals(f)
        ):
            return {
                **scored,
                "complexity_score": min(scored["complexity_score"], max_normal_score),
                "difficulty": "normal",
                "reject_reasons": [],
                "hard_signal_source": "llm_demoted_ordinary",
            }
        return {
            **scored,
            "difficulty": llm_label,
            "reject_reasons": [],
            "hard_signal_source": "llm",
        }

    if scored["is_trivial"] or scored.get("is_too_long"):
        f0 = scored["features"]
        if not (
            f0.get("marathi_digit_count", 0) > 0
            or f0.get("identifier_likeness_score", 0) > 0
        ):
            return {
                **scored,
                "difficulty": "reject",
                "reject_reasons": ["trivial_or_too_long"],
                "hard_signal_source": "none",
            }

    wc = scored["word_count"]
    score = scored["complexity_score"]

    if is_ordinary_prose(text) and normal_min_words <= wc <= normal_max_words:
        return {
            **scored,
            "complexity_score": min(score, max_normal_score),
            "difficulty": "normal",
            "reject_reasons": [],
            "hard_signal_source": "heuristic_ordinary",
        }

    # Hard only with strong generic multi-signals — never slash-alone titles
    if strong_generic_hard_signals(f) and score >= min_hard_score:
        return {
            **scored,
            "difficulty": "hard",
            "reject_reasons": [],
            "hard_signal_source": "heuristic_strong",
        }

    # Weak slash / light punctuation without digits → not hard, not normal
    reasons.append("weak_complexity_awaiting_llm_or_stronger_signals")
    return {
        **scored,
        "difficulty": "reject",
        "reject_reasons": reasons,
        "hard_signal_source": "none",
    }


def apply_llm_complexity(
    rec: dict[str, Any],
    llm_complexity: str | None,
    *,
    text: str | None = None,
    allow_ascii_digits: bool = False,
    min_hard_score: float = 3.0,
) -> dict[str, Any]:
    """Re-label a record using LLM hard|normal|reject (authoritative)."""
    t = text
    if t is None:
        t = rec.get("expected_text") or rec.get("text_assist") or rec.get("ocr_prediction") or ""
    classified = classify_difficulty(
        t,
        allow_ascii_digits=allow_ascii_digits,
        min_hard_score=min_hard_score,
        llm_complexity=llm_complexity,
        score_override=rec.get("complexity_score"),
    )
    rec["difficulty"] = classified["difficulty"]
    rec["complexity_score"] = classified["complexity_score"]
    rec["features"] = classified.get("features") or rec.get("features")
    rec["hard_signal_source"] = classified.get("hard_signal_source")
    rec["contains_marathi_numerals"] = classified.get("contains_marathi_numerals")
    rec["contains_special_chars"] = classified.get("contains_special_chars")
    rec["contains_reference_identifier"] = classified.get("contains_reference_identifier")
    rec["contains_conjuncts"] = classified.get("contains_conjuncts")
    if classified["difficulty"] == "reject":
        rec["reject_reasons"] = list(
            dict.fromkeys(
                (rec.get("reject_reasons") or []) + (classified.get("reject_reasons") or [])
            )
        )
    return rec


def demote_unconfirmed_hard(rec: dict[str, Any]) -> dict[str, Any]:
    """If LLM soft-failed, keep hard only with strong generic signals.

    Prevents title-slash prose from silently filling the hard lane.
    """
    if rec.get("difficulty") != "hard":
        return rec
    src = rec.get("hard_signal_source") or ""
    if src == "llm":
        return rec
    features = rec.get("features") or {}
    if not features:
        text = rec.get("expected_text") or rec.get("text_assist") or ""
        features = compute_features(text)
        rec["features"] = features
    if strong_generic_hard_signals(features):
        rec["hard_signal_source"] = "heuristic_strong_soft_fail"
        return rec
    rec["difficulty"] = "reject"
    rec["reject_reasons"] = list(
        dict.fromkeys(
            (rec.get("reject_reasons") or []) + ["hard_unconfirmed_llm_soft_fail"]
        )
    )
    rec["hard_signal_source"] = "demoted_soft_fail"
    return rec


def score_candidate(
    text: str,
    *,
    min_words: int = 2,
    max_words: int = 50,
    min_chars: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    features = compute_features(text)
    score = complexity_score(text)
    too_long = is_too_long(text, max_words=max_words, max_chars=max_chars)
    trivial = is_trivial(
        text,
        min_words=min_words,
        max_words=max_words,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    n_chars = char_count(text)
    return {
        "complexity_score": score,
        "features": features,
        "word_count": features["word_count"],
        "char_count": n_chars,
        "is_too_long": too_long,
        "is_trivial": trivial or too_long,
        "contains_marathi_numerals": features["marathi_digit_count"] > 0,
        "contains_special_chars": features["symbol_count"] > 0,
        "contains_reference_identifier": features["identifier_likeness_score"] > 0,
        "contains_conjuncts": features["conjunct_estimate"] > 0,
    }
