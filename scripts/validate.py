from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.client import LLMClient
from pipeline.char_limits import (
    resolve_char_limits,
    suggested_min_aspect,
    suggested_min_width,
)
from pipeline.config import load_config, resolve_path
from pipeline.hashing import perceptual_hash_file, sha256_file
from pipeline.io_utils import read_jsonl, write_json, write_jsonl
from pipeline.logging_setup import setup_logging
from pipeline.profiles import get_active_profile
from pipeline.word_limits import resolve_word_limits, suggested_max_crop_height
from validation.duplicates import find_duplicate_images, find_duplicate_texts
from validation.ground_truth_check import ground_truth_checks
from validation.image_quality import check_image_quality

logger = logging.getLogger(__name__)


def _resolve_image(rec: dict) -> Path:
    img = ROOT / rec.get("image_filename", "")
    if img.is_file():
        return img
    return Path(rec.get("image_path", ""))


def _llm_validate_one(client: LLMClient, rec: dict) -> tuple[str, dict]:
    """LLM use #2: vision exact-match of crop image vs candidate text."""
    text = rec.get("expected_text") or rec.get("text_assist") or rec.get("ocr_prediction") or ""
    img = _resolve_image(rec)
    result = client.vision_validate_exact(
        img,
        text,
        ocr_prediction=rec.get("ocr_prediction") or "",
        context={
            "id": rec.get("id"),
            "source_id": rec.get("source_id"),
            "crop_type": rec.get("crop_type"),
            "word_count": rec.get("word_count"),
            "char_count": rec.get("char_count"),
        },
    )
    return str(rec.get("id")), result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate candidates: rules + Kimi vision image↔text exact match"
    )
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None, help="e.g. 50 (default) or 100")
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM vision validation")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg)
    profile = get_active_profile(cfg)
    marathi_only = bool(profile.get("marathi_digits_only", not profile.get("allow_ascii_digits", False)))
    allow_english = bool(profile.get("allow_english_text", False))
    min_words, max_words = resolve_word_limits(cfg, min_words=args.min_words, max_words=args.max_words)
    min_chars, max_chars = resolve_char_limits(cfg, min_chars=args.min_chars, max_chars=args.max_chars)
    logger.info("Word-length filter: %d–%d words (profile=%s)", min_words, max_words, profile.get("_name"))
    if min_chars is not None or max_chars is not None:
        logger.info("Character-length filter: %s–%s NFC chars", min_chars, max_chars)
    max_h = suggested_max_crop_height(max_words, int(cfg.get("max_crop_height", 220)))
    min_w = suggested_min_width(max_chars, int(cfg.get("min_width", 800)))
    min_aspect = suggested_min_aspect(max_chars, float(cfg.get("min_aspect_ratio", 4.0)))
    cand_dir = resolve_path(cfg, "candidates")
    records = read_jsonl(cand_dir / "candidates.jsonl")

    kept = []
    rejected = []
    paths = []
    for rec in records:
        img = _resolve_image(rec)
        text = rec.get("text_assist") or rec.get("expected_text") or ""
        text = text.replace("\u200c", "").replace("\u200d", "")
        visual_only = (rec.get("scoring_mode_used") == "visual") and not text.strip()
        if rec.get("text_assist"):
            rec["expected_text"] = text
        if visual_only:
            # Scanned crop: no PDF text assist — keep for human GT; OCR is pre-label only
            gt = {"ok": True, "issues": [], "notes": "visual_pending_human_gt"}
            rec["expected_text"] = rec.get("expected_text") or ""
        else:
            gt = ground_truth_checks(
                text or (rec.get("ocr_prediction") or ""),
                marathi_digits_only=marathi_only,
                allow_english_text=allow_english,
                min_words=min_words,
                max_words=max_words,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            # Prefer assist text as expected; never promote OCR to GT silently
            if not rec.get("expected_text") and text:
                rec["expected_text"] = text
        iq = check_image_quality(
            img,
            min_width=min_w,
            min_height=int(cfg.get("min_height", 100)),
            min_aspect=min_aspect,
            max_height=max_h + 50,
        )
        rec["validation"] = {"ground_truth": gt, "image_quality": iq}
        if img.is_file():
            rec["sha256"] = sha256_file(img)
            rec["perceptual_hash"] = perceptual_hash_file(img)
            paths.append(str(img))
            rec["image_width"] = iq.get("width", 0)
            rec["image_height"] = iq.get("height", 0)

        reasons = []
        if not gt["ok"]:
            reasons.extend(gt["issues"])
        if not iq["ok"]:
            hard = [x for x in iq["issues"] if x in {"missing_file", "corrupt"} or x.startswith("corrupt")]
            reasons.extend(hard)
        if reasons:
            rec["reject_reasons"] = reasons
            rejected.append(rec)
        else:
            kept.append(rec)

    # Parallel Kimi vision validation (image + text exact match)
    client = LLMClient.from_config(cfg)
    use_llm = client.enabled and not args.skip_llm
    llm_stats = {
        "ran": 0,
        "suspicious": 0,
        "exact_match": 0,
        "errors": 0,
        "vision": 0,
        "text_fallback": 0,
        "skipped": True,
    }
    if use_llm and kept:
        workers = max(1, int(cfg.get("llm_validate_parallel_workers", 2)))
        llm_stats["skipped"] = False
        logger.info(
            "Kimi vision exact-match validation on %d records (workers=%d)",
            len(kept),
            workers,
        )
        by_id: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_llm_validate_one, client, rec) for rec in kept]
            for fut in as_completed(futures):
                rid, llm_result = fut.result()
                by_id[rid] = llm_result
        for rec in kept:
            llm_result = by_id.get(str(rec.get("id")), {"enabled": True, "skipped": True})
            rec.setdefault("validation", {})["llm"] = llm_result
            llm_stats["ran"] += 1
            mode = llm_result.get("mode") or ""
            if mode == "vision":
                llm_stats["vision"] += 1
            elif mode == "text_fallback":
                llm_stats["text_fallback"] += 1
            if llm_result.get("error"):
                llm_stats["errors"] += 1
            if llm_result.get("exact_match"):
                llm_stats["exact_match"] += 1
            if llm_result.get("suspicious") or (
                llm_result.get("exact_match") is False and not llm_result.get("skipped")
            ):
                llm_stats["suspicious"] += 1
                rec["review_status"] = "needs_llm_attention"
            else:
                rec["review_status"] = "pending_review"
    else:
        for rec in kept:
            rec.setdefault("validation", {})["llm"] = {
                "enabled": False,
                "skipped": True,
                "notes": "llm_disabled_or_skip",
                "mode": "disabled",
            }
            rec["review_status"] = "pending_review"

    # automated_qa: image-strict (drop exact/near image dups); text-soft (flag only)
    dups = find_duplicate_images(paths, phash_max_distance=int(cfg.get("perceptual_hash_max_distance", 8)))
    text_dups = find_duplicate_texts(
        kept, text_key="expected_text", threshold=float(cfg.get("text_near_duplicate_threshold", 0.92))
    )
    drop_ids: set = set()
    for group in dups["exact_sha_groups"]:
        for p in group[1:]:
            for r in kept:
                if str(ROOT / r.get("image_filename", "")) == p or r.get("image_path") == p:
                    drop_ids.add(r["id"])
    for pa, pb, _d in dups["near_phash_pairs"]:
        for r in kept:
            rp = str(ROOT / r.get("image_filename", ""))
            if rp == pb or r.get("image_path") == pb:
                drop_ids.add(r["id"])

    text_dup_flags = [
        {"id_a": a, "id_b": b, "similarity": sim} for a, b, sim in text_dups
    ]
    flagged_ids = {str(a) for a, _b, _s in text_dups} | {str(b) for _a, b, _s in text_dups}

    final_kept = []
    for r in kept:
        if r["id"] in drop_ids:
            r["reject_reasons"] = r.get("reject_reasons", []) + ["duplicate_image"]
            rejected.append(r)
        else:
            if str(r.get("id")) in flagged_ids:
                r["text_near_duplicate_flag"] = True
            if r.get("review_status") not in {"needs_llm_attention", "pending_review"}:
                r["review_status"] = "pending_review"
            final_kept.append(r)

    final_kept.sort(key=lambda r: 0 if r.get("review_status") == "needs_llm_attention" else 1)

    write_jsonl(cand_dir / "candidates_valid.jsonl", final_kept)
    write_jsonl(cand_dir / "candidates_rejected.jsonl", rejected)
    write_json(
        cand_dir / "validation_report.json",
        {
            "layer": "automated_qa",
            "profile": profile.get("_name"),
            "kept": len(final_kept),
            "rejected": len(rejected),
            "exact_image_dup_groups": len(dups["exact_sha_groups"]),
            "near_image_pairs": len(dups["near_phash_pairs"]),
            "text_near_dup_flags": len(text_dup_flags),
            "text_near_duplicates": text_dup_flags[:50],
            "llm": llm_stats,
            "min_words": min_words,
            "max_words": max_words,
            "min_chars": min_chars,
            "max_chars": max_chars,
        },
    )
    write_jsonl(cand_dir / "candidates.jsonl", final_kept)
    print(
        f"Validated: kept={len(final_kept)} rejected={len(rejected)} "
        f"llm_exact={llm_stats.get('exact_match', 0)} "
        f"llm_suspicious={llm_stats.get('suspicious', 0)} "
        f"llm_vision={llm_stats.get('vision', 0)} "
        f"llm_errors={llm_stats.get('errors', 0)}"
    )


if __name__ == "__main__":
    main()
