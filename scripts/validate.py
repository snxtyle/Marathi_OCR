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
from scoring.complexity import apply_llm_complexity, demote_unconfirmed_hard
from validation.duplicates import (
    drop_near_duplicate_texts,
    find_duplicate_images,
    find_duplicate_texts,
)
from validation.ground_truth_check import ground_truth_checks
from validation.image_quality import check_image_quality

logger = logging.getLogger(__name__)


def _resolve_image(rec: dict) -> Path:
    img = ROOT / rec.get("image_filename", "")
    if img.is_file():
        return img
    return Path(rec.get("image_path", ""))


def _llm_validate_one(client: LLMClient, rec: dict) -> tuple[str, dict]:
    """LLM end-gate: vision exact-match + complexity judgment."""
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
            "heuristic_difficulty": rec.get("difficulty"),
        },
        intended_lane=str(rec.get("difficulty") or ""),
    )
    return str(rec.get("id")), result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate candidates: rules + Kimi vision image↔text + complexity"
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
    allow_ascii = bool(profile.get("allow_ascii_digits", False))
    marathi_only = bool(profile.get("marathi_digits_only", not allow_ascii))
    allow_english = bool(profile.get("allow_english_text", False))
    min_hard = float(cfg.get("min_complexity_score", 3.0))
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

    client = LLMClient.from_config(cfg)
    use_llm = client.enabled and not args.skip_llm
    llm_stats = {
        "ran": 0,
        "suspicious": 0,
        "exact_match": 0,
        "complex_hard": 0,
        "lane_hard": 0,
        "lane_normal": 0,
        "lane_reject": 0,
        "lane_applied": 0,
        "hard_demoted_soft_fail": 0,
        "errors": 0,
        "vision": 0,
        "text_fallback": 0,
        "skipped": True,
    }
    if use_llm and kept:
        workers = max(1, int(cfg.get("llm_validate_parallel_workers", 2)))
        llm_stats["skipped"] = False
        logger.info(
            "Kimi vision exact-match + complexity validation on %d records (workers=%d)",
            len(kept),
            workers,
        )
        by_id: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_llm_validate_one, client, rec) for rec in kept]
            for fut in as_completed(futures):
                rid, llm_result = fut.result()
                by_id[rid] = llm_result
        still_kept: list[dict] = []
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
            if llm_result.get("is_complex_for_ocr") is True:
                llm_stats["complex_hard"] += 1

            lane = (
                llm_result.get("ocr_complexity")
                or llm_result.get("suggested_lane")
                or ""
            )
            if isinstance(lane, str):
                lane = lane.strip().lower()
            else:
                lane = ""
            if lane == "hard":
                llm_stats["lane_hard"] += 1
            elif lane == "normal":
                llm_stats["lane_normal"] += 1
            elif lane == "reject":
                llm_stats["lane_reject"] += 1

            skipped = bool(llm_result.get("skipped"))
            if not skipped and lane in {"hard", "normal", "reject"}:
                apply_llm_complexity(
                    rec,
                    lane,
                    allow_ascii_digits=allow_ascii,
                    min_hard_score=min_hard,
                )
                llm_stats["lane_applied"] += 1
                rec["llm_suggested_lane"] = lane
                rec["llm_is_complex_for_ocr"] = llm_result.get("is_complex_for_ocr")
            elif skipped:
                # Soft-fail: do not silently keep weak hard (title-slash prose)
                before = rec.get("difficulty")
                demote_unconfirmed_hard(rec)
                if before == "hard" and rec.get("difficulty") != "hard":
                    llm_stats["hard_demoted_soft_fail"] += 1

            if rec.get("difficulty") == "reject":
                rejected.append(rec)
                continue

            mismatch = llm_result.get("exact_match") is False and not skipped
            not_complex_for_hard = (
                (rec.get("difficulty") == "hard")
                and llm_result.get("is_complex_for_ocr") is False
                and not skipped
            )
            lane_reject = lane == "reject" and not skipped
            if llm_result.get("suspicious") or mismatch or not_complex_for_hard or lane_reject:
                llm_stats["suspicious"] += 1
                rec["review_status"] = "needs_llm_attention"
            else:
                rec["review_status"] = "pending_review"
            still_kept.append(rec)
        kept = still_kept
    else:
        still_kept = []
        for rec in kept:
            rec.setdefault("validation", {})["llm"] = {
                "enabled": False,
                "skipped": True,
                "notes": "llm_disabled_or_skip",
                "mode": "disabled",
                "is_complex_for_ocr": None,
                "suggested_lane": None,
                "ocr_complexity": None,
            }
            before = rec.get("difficulty")
            demote_unconfirmed_hard(rec)
            if before == "hard" and rec.get("difficulty") != "hard":
                llm_stats["hard_demoted_soft_fail"] += 1
                rejected.append(rec)
                continue
            rec["review_status"] = "pending_review"
            still_kept.append(rec)
        kept = still_kept

    # Image-strict dedup; text near-dup / prefix twins DROP (not flag-only)
    dups = find_duplicate_images(paths, phash_max_distance=int(cfg.get("perceptual_hash_max_distance", 8)))
    text_thresh = float(cfg.get("text_near_duplicate_threshold", 0.92))
    text_dups = find_duplicate_texts(kept, text_key="expected_text", threshold=text_thresh)
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

    after_img = []
    for r in kept:
        if r["id"] in drop_ids:
            r["reject_reasons"] = r.get("reject_reasons", []) + ["duplicate_image"]
            rejected.append(r)
        else:
            after_img.append(r)

    final_kept, text_drop_stats = drop_near_duplicate_texts(
        after_img,
        text_key="expected_text",
        threshold=max(0.88, text_thresh - 0.04),
        prefer_key="complexity_score",
        cross_source=True,
    )
    dropped_text_ids = {str(r.get("id")) for r in after_img} - {str(r.get("id")) for r in final_kept}
    for r in after_img:
        if str(r.get("id")) in dropped_text_ids:
            r["reject_reasons"] = r.get("reject_reasons", []) + ["text_near_duplicate"]
            rejected.append(r)

    text_dup_flags = [{"id_a": a, "id_b": b, "similarity": sim} for a, b, sim in text_dups]

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
            "text_near_dup_dropped": text_drop_stats.get("dropped", 0),
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
        f"llm_lane_applied={llm_stats.get('lane_applied', 0)} "
        f"hard_demoted={llm_stats.get('hard_demoted_soft_fail', 0)} "
        f"text_near_dropped={text_drop_stats.get('dropped', 0)}"
    )


if __name__ == "__main__":
    main()
