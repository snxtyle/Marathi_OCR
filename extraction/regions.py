from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from pipeline.char_limits import char_count
from scoring.complexity import score_candidate
from validation.numerals import contains_ascii_digits


logger = logging.getLogger(__name__)

DISCOVERY_DPI = 180

# Allow Devanagari + Marathi digits + common gov-doc punctuation only.
_ALLOWED_EXTRA = set("/.,-—–:;()[]*%₹«»“”|'\"+=_")


def _text_encoding_ok(text: str) -> bool:
    """Reject private-use / Latin-extended junk common in broken PDF CMaps."""
    for ch in text:
        if ch.isspace() or ch in _ALLOWED_EXTRA:
            continue
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:  # Devanagari block (incl. digits)
            continue
        if ch in "०१२३४५६७८९":
            continue
        # ASCII digits should already be converted; leftover digits fail digits-only later
        if ch.isascii() and not ch.isalpha():
            continue
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            return False
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        return False
    return True


def _normalize_assist(text: str, *, allow_ascii_digits: bool = False) -> str | None:
    """Normalize whitespace; reject ASCII digits unless profile allows them."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.replace("\u200c", "").replace("\u200d", "")
    if not allow_ascii_digits and contains_ascii_digits(text):
        return None
    return text


def _line_from_words(
    words: list[tuple],
    *,
    allow_ascii_digits: bool = False,
) -> dict[str, Any] | None:
    """words: (x0,y0,x1,y1,text, block, line, word)"""
    if not words:
        return None
    xs0 = [w[0] for w in words]
    ys0 = [w[1] for w in words]
    xs1 = [w[2] for w in words]
    ys1 = [w[3] for w in words]
    text = _normalize_assist(" ".join(w[4] for w in words), allow_ascii_digits=allow_ascii_digits)
    if not text:
        return None
    return {
        "text": text,
        "bbox_180": [min(xs0), min(ys0), max(xs1), max(ys1)],
        "line_height_180": max(ys1) - min(ys0),
        "words": words,
    }


def _words_180dpi(page: fitz.Page) -> list[tuple]:
    """PDF word tuples with xy converted from points to 180-DPI pixels."""
    scale = DISCOVERY_DPI / 72.0
    out: list[tuple] = []
    for w in page.get_text("words"):
        x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
        out.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale, w[4], w[5], w[6], w[7]))
    return out


def extract_lines_from_page(
    page: fitz.Page,
    *,
    allow_ascii_digits: bool = False,
) -> list[dict[str, Any]]:
    words = _words_180dpi(page)
    if not words:
        return []
    # Group by (block, line)
    groups: dict[tuple[int, int], list] = {}
    for w in words:
        key = (int(w[5]), int(w[6]))
        groups.setdefault(key, []).append(w)
    lines: list[dict[str, Any]] = []
    for key in sorted(groups.keys()):
        line = _line_from_words(
            sorted(groups[key], key=lambda x: x[0]),
            allow_ascii_digits=allow_ascii_digits,
        )
        if line:
            lines.append(line)
    return lines


def classify_region(text: str) -> str:
    if re.search(r"प्र\.?\s*क्र|जा\.?\s*क्र|क्रमांक|शासन\s*निर्णय|अधिसूचना", text):
        return "reference_identifier"
    if re.search(r"दिनांक|\bदि\.", text):
        return "date_line"
    if re.search(r"रु\.|₹|%", text):
        return "financial"
    if re.match(r"^[०-९]+[.)]", text):
        return "numbered_section"
    if len(text) > 80:
        return "paragraph_section"
    return "text_line"


def _word_span_segments(
    words: list[tuple],
    *,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Contiguous word runs whose NFC stripped length is in [min_chars, max_chars]."""
    segs: list[dict[str, Any]] = []
    n = len(words)
    for i in range(n):
        for j in range(i, n):
            chunk = words[i : j + 1]
            text = _normalize_assist(" ".join(w[4] for w in chunk))
            c = char_count(text)
            if c > max_chars:
                break
            if c < min_chars:
                continue
            xs0 = [w[0] for w in chunk]
            ys0 = [w[1] for w in chunk]
            xs1 = [w[2] for w in chunk]
            ys1 = [w[3] for w in chunk]
            segs.append(
                {
                    "text": text,
                    "bbox_180": [min(xs0), min(ys0), max(xs1), max(ys1)],
                    "line_height_180": max(ys1) - min(ys0),
                    "span_i": i,
                    "span_j": j,
                }
            )
    return segs


def extract_visual_bands_from_page(
    page: fitz.Page,
    *,
    page_index: int,
    source_id: str,
    pdf_path: Path,
    source_url: str = "",
    document_type: str = "unknown",
    source_document: str = "",
    max_bands: int = 8,
) -> list[dict[str, Any]]:
    """Layout/geometry bands for scanned pages with no reliable PDF text layer."""
    from scoring.complexity import visual_complexity_score

    pix = page.get_pixmap(dpi=DISCOVERY_DPI, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
    w, h = img.size
    if w < 100 or h < 100:
        return []
    # Row ink density
    pixels = img.load()
    row_ink = []
    for y in range(h):
        ink = 0
        for x in range(0, w, 3):  # subsample x
            if pixels[x, y] < 200:
                ink += 1
        row_ink.append(ink)
    thresh = max(8, int(0.02 * (w / 3)))
    bands: list[tuple[int, int]] = []
    y = 0
    while y < h:
        while y < h and row_ink[y] < thresh:
            y += 1
        if y >= h:
            break
        y0 = y
        while y < h and row_ink[y] >= thresh:
            y += 1
        y1 = y
        band_h = y1 - y0
        if 28 <= band_h <= 160:
            bands.append((y0, y1))
        elif band_h > 160:
            # Split tall blocks into line-ish windows
            step = 70
            for yy in range(y0, y1 - 40, step):
                bands.append((yy, min(y1, yy + 90)))
    # Prefer mid-page content bands
    scored_bands: list[tuple[float, tuple[int, int]]] = []
    for y0, y1 in bands:
        band_h = y1 - y0
        aspect = w / max(band_h, 1)
        vs = visual_complexity_score(width=w, height=band_h, aspect_ratio=aspect)
        # Prefer wider aspect (single line-ish)
        scored_bands.append((vs + min(aspect / 10.0, 1.5), (y0, y1)))
    scored_bands.sort(key=lambda x: x[0], reverse=True)

    out: list[dict[str, Any]] = []
    margin_x = int(0.05 * w)
    for i, (vs, (y0, y1)) in enumerate(scored_bands[:max_bands]):
        bbox = [float(margin_x), float(y0), float(w - margin_x), float(y1)]
        band_h = y1 - y0
        aspect = (bbox[2] - bbox[0]) / max(band_h, 1)
        cand_id = f"{source_id}_p{page_index + 1:03d}_v{i:04d}"
        out.append(
            {
                "id": cand_id,
                "source_id": source_id,
                "source_path": str(pdf_path),
                "source_document": source_document or pdf_path.name,
                "source_url": source_url,
                "document_type": document_type,
                "source_page": page_index + 1,
                "text_assist": "",
                "bbox_discovery_180dpi": bbox,
                "original_coordinates": {"dpi": DISCOVERY_DPI, "bbox": bbox},
                "line_height_180": float(band_h),
                "crop_type": "visual_band",
                "difficulty": "hard" if vs >= 3.0 else "normal",
                "word_count": 0,
                "char_count": 0,
                "complexity_score": vs,
                "contains_marathi_numerals": False,
                "contains_special_chars": False,
                "contains_reference_identifier": False,
                "contains_conjuncts": False,
                "features": {"visual_score": vs, "aspect_ratio": aspect},
                "scoring_mode_used": "visual",
            }
        )
    return out


def extract_candidates_from_pdf(
    pdf_path: str | Path,
    *,
    source_id: str,
    source_url: str = "",
    document_type: str = "unknown",
    source_document: str = "",
    min_complexity: float = 0.0,
    max_line_height_180: float = 72.0,
    min_words: int = 2,
    max_words: int = 50,
    min_chars: int | None = None,
    max_chars: int | None = None,
    allow_ascii_digits: bool = False,
    dual_lane: bool = True,
) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    candidates: list[dict[str, Any]] = []
    use_spans = min_chars is not None and max_chars is not None
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_before = len(candidates)
        for i, line in enumerate(
            extract_lines_from_page(page, allow_ascii_digits=allow_ascii_digits)
        ):
            if line["line_height_180"] > max_line_height_180:
                continue
            if use_spans:
                units = _word_span_segments(
                    line["words"],
                    min_chars=int(min_chars),
                    max_chars=int(max_chars),
                )
            else:
                units = [
                    {
                        "text": line["text"],
                        "bbox_180": line["bbox_180"],
                        "line_height_180": line["line_height_180"],
                        "span_i": 0,
                        "span_j": -1,
                    }
                ]
            ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for unit in units:
                text = unit["text"]
                if not allow_ascii_digits and contains_ascii_digits(text):
                    continue
                text_norm = _normalize_assist(text, allow_ascii_digits=allow_ascii_digits)
                if not text_norm:
                    continue
                text = text_norm
                if not _text_encoding_ok(text):
                    continue
                if re.search(r"[A-Za-z]", text):
                    continue
                from scoring.complexity import classify_difficulty

                if dual_lane:
                    classified = classify_difficulty(
                        text,
                        min_hard_score=min_complexity if min_complexity > 0 else 3.0,
                        allow_ascii_digits=allow_ascii_digits,
                        min_words=min_words,
                        max_words=max_words,
                        min_chars=min_chars,
                        max_chars=max_chars,
                    )
                    if classified["difficulty"] == "reject":
                        continue
                    scored = classified
                else:
                    scored = score_candidate(
                        text,
                        min_words=min_words,
                        max_words=max_words,
                        min_chars=min_chars,
                        max_chars=max_chars,
                    )
                    if scored["is_trivial"] or scored.get("is_too_long"):
                        continue
                    if scored["complexity_score"] < min_complexity:
                        continue
                    scored = {**scored, "difficulty": "hard", "reject_reasons": []}
                ranked.append((scored["complexity_score"], unit, scored))
            ranked.sort(key=lambda x: x[0], reverse=True)
            seen_line_text: set[str] = set()
            line_kept = 0
            per_line_cap = 3 if use_spans else 1
            for u_idx, (_score, unit, scored) in enumerate(ranked):
                text = unit["text"]
                text_norm = _normalize_assist(text, allow_ascii_digits=allow_ascii_digits)
                if not text_norm:
                    continue
                text = text_norm
                if text in seen_line_text:
                    continue
                seen_line_text.add(text)
                span_tag = f"s{unit.get('span_i', 0)}-{unit.get('span_j', 0)}"
                cand_id = f"{source_id}_p{page_index + 1:03d}_l{i:04d}_{span_tag}_{u_idx}"
                candidates.append(
                    {
                        "id": cand_id,
                        "source_id": source_id,
                        "source_path": str(pdf_path),
                        "source_document": source_document or pdf_path.name,
                        "source_url": source_url,
                        "document_type": document_type,
                        "source_page": page_index + 1,
                        "text_assist": text,
                        "bbox_discovery_180dpi": unit["bbox_180"],
                        "original_coordinates": {
                            "dpi": DISCOVERY_DPI,
                            "bbox": unit["bbox_180"],
                        },
                        "line_height_180": unit["line_height_180"],
                        "crop_type": classify_region(text),
                        "difficulty": scored.get("difficulty", "hard"),
                        "word_count": scored["word_count"],
                        "char_count": scored["char_count"],
                        **{
                            k: scored[k]
                            for k in (
                                "complexity_score",
                                "contains_marathi_numerals",
                                "contains_special_chars",
                                "contains_reference_identifier",
                                "contains_conjuncts",
                            )
                        },
                        "features": scored["features"],
                        "scoring_mode_used": "text",
                    }
                )
                line_kept += 1
                if line_kept >= per_line_cap:
                    break
        # Scanned / empty text layer → visual bands
        if len(candidates) == page_before:
            candidates.extend(
                extract_visual_bands_from_page(
                    page,
                    page_index=page_index,
                    source_id=source_id,
                    pdf_path=pdf_path,
                    source_url=source_url,
                    document_type=document_type,
                    source_document=source_document,
                )
            )
    doc.close()
    candidates.sort(key=lambda c: c["complexity_score"], reverse=True)
    logger.info("Extracted %d candidates from %s", len(candidates), pdf_path.name)
    return candidates


def render_candidate_crop(
    pdf_path: str | Path,
    page_number: int,
    bbox_180: list[float],
    out_path: str | Path,
    *,
    dpi: int = 600,
    discovery_dpi: int = DISCOVERY_DPI,
    margin_x: int = 28,
    margin_y: int = 8,
) -> dict[str, Any]:
    scale = dpi / discovery_dpi
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    doc.close()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    x0, y0, x1, y1 = bbox_180
    left = max(0, round(x0 * scale) - margin_x)
    top = max(0, round(y0 * scale) - margin_y)
    right = min(img.width, round(x1 * scale) + margin_x)
    bottom = min(img.height, round(y1 * scale) + margin_y)
    crop = img.crop((left, top, right, bottom))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, format="PNG", dpi=(dpi, dpi))
    return {
        "image_path": str(out_path),
        "image_width": crop.width,
        "image_height": crop.height,
        "dpi": dpi,
        "bbox_render": [left, top, right - left, bottom - top],
    }
