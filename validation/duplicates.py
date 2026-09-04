from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable

from pipeline.hashing import perceptual_hash_distance, perceptual_hash_file, sha256_file


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = re.sub(r"\s+", "", text)
    return text


def text_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_duplicate_images(
    paths: Iterable[str],
    *,
    phash_max_distance: int = 8,
) -> dict[str, Any]:
    paths = list(paths)
    sha_map: dict[str, list[str]] = {}
    phashes: list[tuple[str, str]] = []
    for p in paths:
        try:
            s = sha256_file(p)
            sha_map.setdefault(s, []).append(p)
            phashes.append((p, perceptual_hash_file(p)))
        except OSError:
            continue

    exact = [group for group in sha_map.values() if len(group) > 1]
    near: list[tuple[str, str, int]] = []
    for i, (pa, ha) in enumerate(phashes):
        for pb, hb in phashes[:i]:
            dist = perceptual_hash_distance(ha, hb)
            if dist <= phash_max_distance and pa != pb:
                near.append((pa, pb, dist))
    return {"exact_sha_groups": exact, "near_phash_pairs": near}


def find_duplicate_texts(
    records: list[dict[str, Any]],
    text_key: str = "expected_text",
    *,
    threshold: float = 0.92,
) -> list[tuple[Any, Any, float]]:
    pairs: list[tuple[Any, Any, float]] = []
    texts = [(r.get("id", i), r.get(text_key, "")) for i, r in enumerate(records)]
    for i, (ida, ta) in enumerate(texts):
        for idb, tb in texts[:i]:
            sim = text_similarity(ta, tb)
            if sim >= threshold:
                pairs.append((idb, ida, round(sim, 4)))
    return pairs


def candidate_dedup_before_ocr(
    records: list[dict[str, Any]],
    *,
    phash_max_distance: int = 8,
    bbox_iou_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stage A: drop same region / exact image / near-phash duplicates before OCR."""
    kept: list[dict[str, Any]] = []
    drop_ids: set[str] = set()
    stats = {
        "input": len(records),
        "exact_image": 0,
        "near_image": 0,
        "same_bbox": 0,
        "kept": 0,
    }

    # Same source page + high IoU bbox
    for i, a in enumerate(records):
        if a.get("id") in drop_ids:
            continue
        ab = a.get("bbox_discovery_180dpi") or (a.get("original_coordinates") or {}).get("bbox")
        for b in records[:i]:
            if b.get("id") in drop_ids:
                continue
            if a.get("source_id") != b.get("source_id"):
                continue
            if a.get("source_page") != b.get("source_page"):
                continue
            bb = b.get("bbox_discovery_180dpi") or (b.get("original_coordinates") or {}).get("bbox")
            if ab and bb and _bbox_iou(list(ab), list(bb)) >= bbox_iou_threshold:
                drop_ids.add(str(a.get("id")))
                stats["same_bbox"] += 1
                break

    # Image sha / phash among records that already have files
    path_by_id: dict[str, str] = {}
    for r in records:
        rid = str(r.get("id"))
        if rid in drop_ids:
            continue
        p = r.get("image_path") or ""
        if not p:
            # relative image_filename may exist after crop
            from pathlib import Path

            root = Path(__file__).resolve().parents[1]
            cand = root / (r.get("image_filename") or "")
            if cand.is_file():
                p = str(cand)
        if p:
            path_by_id[rid] = p

    if path_by_id:
        dups = find_duplicate_images(path_by_id.values(), phash_max_distance=phash_max_distance)
        path_to_ids: dict[str, list[str]] = {}
        for rid, p in path_by_id.items():
            path_to_ids.setdefault(p, []).append(rid)
        for group in dups["exact_sha_groups"]:
            ids = []
            for p in group:
                ids.extend(path_to_ids.get(p, []))
            for rid in ids[1:]:
                if rid not in drop_ids:
                    drop_ids.add(rid)
                    stats["exact_image"] += 1
        for pa, pb, _d in dups["near_phash_pairs"]:
            for rid in path_to_ids.get(pb, [])[1:] or path_to_ids.get(pb, []):
                # drop the later of the pair
                pass
            ids_a = path_to_ids.get(pa, [])
            ids_b = path_to_ids.get(pb, [])
            for rid in ids_b:
                if rid not in drop_ids and rid not in ids_a[:1]:
                    drop_ids.add(rid)
                    stats["near_image"] += 1

    for r in records:
        if str(r.get("id")) in drop_ids:
            continue
        kept.append(r)
    stats["kept"] = len(kept)
    return kept, stats


def post_review_text_dedup(
    records: list[dict[str, Any]],
    *,
    text_key: str = "expected_text",
    threshold: float = 0.92,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stage B: flag near-duplicate text; reject only if same source family (no new visual value)."""
    pairs = find_duplicate_texts(records, text_key=text_key, threshold=threshold)
    by_id = {str(r.get("id")): r for r in records}
    flagged: list[dict[str, Any]] = []
    drop: set[str] = set()
    for ida, idb, sim in pairs:
        a, b = by_id.get(str(ida)), by_id.get(str(idb))
        if not a or not b:
            continue
        entry = {
            "id_a": ida,
            "id_b": idb,
            "similarity": sim,
            "same_source": a.get("source_id") == b.get("source_id"),
        }
        flagged.append(entry)
        # Reject only when same source_id (header/footer repeats) — keep cross-doc same text
        if a.get("source_id") == b.get("source_id"):
            drop.add(str(idb))
    kept = [r for r in records if str(r.get("id")) not in drop]
    return kept, flagged
