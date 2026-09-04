"""OCR bakeoff metrics for hard Marathi crops.

Ground truth is human/curated ``expected_text`` only — never OCR output.
Default comparison normalization is NFC + strip ZWJ/ZWNJ + collapse whitespace;
strict variants keep the raw strings (aside from optional outer strip).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

from rapidfuzz.distance import Levenshtein

from pipeline.char_limits import ZWJ, ZWNJ, char_count
from scoring.reference_detector import contains_reference_identifier
from validation.numerals import has_ascii_digits, has_marathi_digits

MARATHI_DIGITS = set("०१२३४५६७८९")
ASCII_DIGITS = set("0123456789")

# Punctuation / special set used for accuracy + F1
SPECIAL = set("/.,-—–:;()[]{}*%₹«»“”|")

# Named hard clusters + virama-based detection
CONJUNCT_CLUSTERS = (
    "क्ष",
    "त्र",
    "ज्ञ",
    "श्र",
    "द्व",
    "त्त",
    "द्ध",
    "क्त",
    "ण्य",
    "ल्ल",
    "प्र",
    "र्",
)
VIRAMA = "\u094d"
# Consonant + virama + consonant (covers many rare aksharas)
_VIRAMA_CLUSTER_RE = re.compile(
    r"[\u0900-\u097F]" + VIRAMA + r"[\u0900-\u097F]"
)

# Reference-like: slashes, प्र.क्र., dates, का.१४ style, detector patterns
REF_SPAN_RE = re.compile(
    r"(प्र\.?\s*क्र|जा\.?\s*क्र|क्रमांक|का\.\s*[०-९]+|"
    r"[०-९A-Za-zअ-ह\.]{1,}/[०-९A-Za-zअ-ह\./-]{1,})"
)
DATE_LIKE_RE = re.compile(r"(दिनांक|दि\.\s*[०-९]|[०-९]{1,2}[./-][०-९]{1,2}[./-][०-९]{2,4})")

SHORT_CHAR_BAND = 25  # ≤25 Unicode chars (char_count) vs longer


def normalize_strict(text: str) -> str:
    """As-is comparison: only strip outer whitespace."""
    return (text or "").strip()


def normalize_for_compare(text: str) -> str:
    """NFC, drop ZWJ/ZWNJ, collapse whitespace."""
    text = (text or "").replace(ZWJ, "").replace(ZWNJ, "")
    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _cer_raw(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def cer_strict(ref: str, hyp: str) -> float:
    return _cer_raw(normalize_strict(ref), normalize_strict(hyp))


def cer_normalized(ref: str, hyp: str) -> float:
    return _cer_raw(normalize_for_compare(ref), normalize_for_compare(hyp))


def cer(ref: str, hyp: str) -> float:
    """Backward-compatible alias: normalized CER."""
    return cer_normalized(ref, hyp)


def wer_tokens(ref: str, hyp: str) -> float:
    """Word error rate via token-level Levenshtein / len(gt_tokens)."""
    ref_w = normalize_for_compare(ref).split()
    hyp_w = normalize_for_compare(hyp).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    n, m = len(ref_w), len(hyp_w)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def exact_match(ref: str, hyp: str, *, strict: bool = False) -> bool:
    if strict:
        return normalize_strict(ref) == normalize_strict(hyp)
    return normalize_for_compare(ref) == normalize_for_compare(hyp)


def _extract_chars(text: str, alphabet: set[str]) -> str:
    return "".join(c for c in text if c in alphabet)


def subset_accuracy(ref: str, hyp: str, alphabet: set[str]) -> float:
    r = _extract_chars(normalize_for_compare(ref), alphabet)
    h = _extract_chars(normalize_for_compare(hyp), alphabet)
    if not r:
        return 1.0 if not h else 0.0
    return 1.0 - (Levenshtein.distance(r, h) / len(r))


def punctuation_f1(ref: str, hyp: str, alphabet: set[str] | None = None) -> dict[str, float]:
    """Multiset precision / recall / F1 over the special-char alphabet."""
    alphabet = alphabet or SPECIAL
    rc = Counter(c for c in normalize_for_compare(ref) if c in alphabet)
    hc = Counter(c for c in normalize_for_compare(hyp) if c in alphabet)
    gt_n = sum(rc.values())
    pred_n = sum(hc.values())
    if gt_n == 0 and pred_n == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = sum((rc & hc).values())
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / gt_n if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def is_reference_like(text: str) -> bool:
    t = text or ""
    if contains_reference_identifier(t):
        return True
    if REF_SPAN_RE.search(t):
        return True
    if DATE_LIKE_RE.search(t):
        return True
    if t.count("/") >= 2:
        return True
    return False


def reference_id_exact(ref: str, hyp: str) -> float | None:
    """Full-string exact match on reference-like GT; None if N/A."""
    if not is_reference_like(ref):
        return None
    return 1.0 if exact_match(ref, hyp) else 0.0


def ascii_digit_contamination(ref: str, hyp: str) -> dict[str, Any]:
    """When GT has Marathi digits, flag ASCII 0-9 appearing in the prediction."""
    gt_has = has_marathi_digits(ref)
    ascii_in_pred = has_ascii_digits(hyp) if gt_has else False
    ascii_count = sum(1 for c in (hyp or "") if c in ASCII_DIGITS) if gt_has else 0
    return {
        "gt_has_marathi_digits": gt_has,
        "ascii_digit_contamination": bool(ascii_in_pred),
        "ascii_digit_count_in_pred": ascii_count,
    }


def find_conjunct_occurrences(text: str) -> list[str]:
    """Named clusters + virama C+्+C spans present in text (NFC)."""
    text = unicodedata.normalize("NFC", text or "")
    found: list[str] = []
    covered = [False] * len(text)

    for cluster in sorted(CONJUNCT_CLUSTERS, key=len, reverse=True):
        start = 0
        while True:
            i = text.find(cluster, start)
            if i < 0:
                break
            end = i + len(cluster)
            if not any(covered[i:end]):
                found.append(cluster)
                for j in range(i, end):
                    covered[j] = True
            start = i + 1

    for m in _VIRAMA_CLUSTER_RE.finditer(text):
        i, end = m.start(), m.end()
        if any(covered[i:end]):
            continue
        found.append(m.group(0))
        for j in range(i, end):
            covered[j] = True
    return found


def conjunct_error_rate(ref: str, hyp: str) -> float | None:
    """Fraction of GT conjunct occurrences missing/under-counted in hyp."""
    gt = Counter(find_conjunct_occurrences(ref))
    if not gt:
        return None
    hyp_c = Counter(find_conjunct_occurrences(hyp))
    missing = sum(max(0, gt[k] - hyp_c.get(k, 0)) for k in gt)
    return missing / sum(gt.values())


def length_bands(ref: str) -> dict[str, str]:
    n_chars = char_count(ref)
    n_words = len(normalize_for_compare(ref).split())
    char_band = "short_le_25" if n_chars <= SHORT_CHAR_BAND else "long_gt_25"
    if n_words <= 1:
        word_band = "words_1"
    elif n_words <= 5:
        word_band = "words_2_5"
    else:
        word_band = "words_6_plus"
    return {
        "char_count": n_chars,
        "word_count": n_words,
        "char_band": char_band,
        "word_band": word_band,
    }


def score_pair(ref: str, hyp: str, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-sample metrics. ``meta`` may include issue_type / source_id."""
    meta = meta or {}
    punct = punctuation_f1(ref, hyp)
    contam = ascii_digit_contamination(ref, hyp)
    bands = length_bands(ref)
    conj = conjunct_error_rate(ref, hyp)
    return {
        "cer": cer_normalized(ref, hyp),
        "cer_strict": cer_strict(ref, hyp),
        "cer_normalized": cer_normalized(ref, hyp),
        "wer": wer_tokens(ref, hyp),
        "exact_match": exact_match(ref, hyp, strict=False),
        "exact_match_strict": exact_match(ref, hyp, strict=True),
        "marathi_numeral_accuracy": subset_accuracy(ref, hyp, MARATHI_DIGITS),
        "ascii_digit_contamination": contam["ascii_digit_contamination"],
        "gt_has_marathi_digits": contam["gt_has_marathi_digits"],
        "ascii_digit_count_in_pred": contam["ascii_digit_count_in_pred"],
        "special_char_accuracy": subset_accuracy(ref, hyp, SPECIAL),
        "special_char_precision": punct["precision"],
        "special_char_recall": punct["recall"],
        "special_char_f1": punct["f1"],
        "reference_id_exact": reference_id_exact(ref, hyp),
        "is_reference_like": is_reference_like(ref),
        "conjunct_error_rate": conj,
        "conjunct_applicable": conj is not None,
        **bands,
        "issue_type": meta.get("issue_type") or "",
        "source_id": meta.get("source_id") or "",
    }


def _mean(vals: Iterable[float]) -> float | None:
    vals = list(vals)
    if not vals:
        return None
    return sum(vals) / len(vals)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(xs[lo])
    frac = rank - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def _core_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    ref_vals = [r["reference_id_exact"] for r in rows if r.get("reference_id_exact") is not None]
    conj_vals = [r["conjunct_error_rate"] for r in rows if r.get("conjunct_error_rate") is not None]
    digit_rows = [r for r in rows if r.get("gt_has_marathi_digits")]
    latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    return {
        "n": n,
        "cer": _mean(r["cer"] for r in rows),
        "cer_strict": _mean(r["cer_strict"] for r in rows),
        "cer_normalized": _mean(r["cer_normalized"] for r in rows),
        "wer": _mean(r["wer"] for r in rows),
        "exact_match": sum(1 for r in rows if r.get("exact_match")) / n,
        "exact_match_strict": sum(1 for r in rows if r.get("exact_match_strict")) / n,
        "marathi_numeral_accuracy": _mean(r["marathi_numeral_accuracy"] for r in rows),
        "ascii_digit_contamination_rate": (
            sum(1 for r in digit_rows if r.get("ascii_digit_contamination")) / len(digit_rows)
            if digit_rows
            else None
        ),
        "marathi_digit_samples": len(digit_rows),
        "special_char_accuracy": _mean(r["special_char_accuracy"] for r in rows),
        "special_char_f1": _mean(r["special_char_f1"] for r in rows),
        "special_char_precision": _mean(r["special_char_precision"] for r in rows),
        "special_char_recall": _mean(r["special_char_recall"] for r in rows),
        "reference_id_exact_match": (sum(ref_vals) / len(ref_vals)) if ref_vals else None,
        "reference_id_applicable": len(ref_vals),
        "conjunct_error_rate": _mean(conj_vals) if conj_vals else None,
        "conjunct_applicable": len(conj_vals),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_samples": len(latencies),
    }


def _slice_means(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        label = r.get(key) or "unknown"
        buckets[str(label)].append(r)
    out: dict[str, Any] = {}
    for label, group in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        m = _core_means(group)
        out[label] = {
            "n": m["n"],
            "cer": m["cer"],
            "cer_strict": m["cer_strict"],
            "cer_normalized": m["cer_normalized"],
            "wer": m["wer"],
            "exact_match": m["exact_match"],
            "marathi_numeral_accuracy": m["marathi_numeral_accuracy"],
            "special_char_f1": m["special_char_f1"],
            "reference_id_exact_match": m["reference_id_exact_match"],
            "conjunct_error_rate": m["conjunct_error_rate"],
        }
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall metrics + length bands + report slices."""
    if not rows:
        return {}
    summary = _core_means(rows)

    by_char_band = _slice_means(rows, "char_band")
    by_word_band = _slice_means(rows, "word_band")
    summary["length_band_cer"] = {
        band: {"n": v["n"], "cer": v["cer"], "cer_strict": v["cer_strict"], "cer_normalized": v["cer_normalized"]}
        for band, v in by_char_band.items()
    }
    summary["word_band_cer"] = {
        band: {"n": v["n"], "cer": v["cer"], "wer": v["wer"]} for band, v in by_word_band.items()
    }

    # contains_marathi_digits slice (boolean key as string)
    digit_keyed = []
    for r in rows:
        rr = dict(r)
        rr["contains_marathi_digits"] = "true" if r.get("gt_has_marathi_digits") else "false"
        digit_keyed.append(rr)

    summary["slices"] = {
        "by_issue_type": _slice_means(rows, "issue_type"),
        "by_contains_marathi_digits": _slice_means(digit_keyed, "contains_marathi_digits"),
        "by_source_id": _aggregate_source_id(rows),
        "by_char_band": by_char_band,
        "by_word_band": by_word_band,
    }
    return summary


def _aggregate_source_id(rows: list[dict[str, Any]], *, max_sources: int = 40) -> dict[str, Any]:
    """Compact per-source aggregates (no per-row leakage)."""
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        sid = (r.get("source_id") or "").strip() or "unknown"
        buckets[sid].append(r)
    ranked = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shown = ranked[:max_sources]
    omitted = max(0, len(ranked) - max_sources)
    out: dict[str, Any] = {
        "n_sources": len(ranked),
        "omitted_sources": omitted,
        "sources": {},
    }
    for sid, group in shown:
        m = _core_means(group)
        out["sources"][sid] = {
            "n": m["n"],
            "cer": m["cer"],
            "exact_match": m["exact_match"],
            "wer": m["wer"],
        }
    return out
