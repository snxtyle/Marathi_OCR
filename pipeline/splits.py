from __future__ import annotations

import re
from typing import Any


def source_level_split(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split by source_id to prevent page-level leakage across train/val."""
    import random

    sources = sorted({r["source_id"] for r in records if r.get("source_id")})
    rng = random.Random(seed)
    rng.shuffle(sources)
    cut = max(1, int(len(sources) * train_ratio)) if len(sources) > 1 else len(sources)
    train_sources = set(sources[:cut])
    val_sources = set(sources[cut:]) if cut < len(sources) else set()
    if not val_sources and len(sources) > 1:
        val_sources = {sources[-1]}
        train_sources.discard(sources[-1])
    train = [r for r in records if r.get("source_id") in train_sources]
    val = [r for r in records if r.get("source_id") in val_sources]
    return {"train": train, "val": val, "train_sources": sorted(train_sources), "val_sources": sorted(val_sources)}


ISSUE_TYPES = [
    "complex_reference_number",
    "marathi_numerals",
    "special_chars_issue",
    "date_issue",
    "conjunct_character_issue",
    "matra_issue",
    "dense_document_text",
    "government_document",
    "legal_text",
    "header_footer",
    "document_identifier",
]


def classify_issue_type(text: str, crop_type: str = "") -> str:
    if re.search(r"प्र\.?\s*क्र|जा\.?\s*क्र|क्रमांक|शासन\s*निर्णय", text):
        return "complex_reference_number"
    if re.search(r"[०-९].*[०-९]", text) and ("/" in text or "." in text):
        return "marathi_numerals"
    if any(c in text for c in "/₹%()[]—"):
        return "special_chars_issue"
    if "दि." in text or "दिनांक" in text:
        return "date_issue"
    if text.count("\u094d") >= 3:
        return "conjunct_character_issue"
    if crop_type == "legal_text":
        return "legal_text"
    return "dense_document_text"
