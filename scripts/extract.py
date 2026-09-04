from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction.regions import extract_candidates_from_pdf, render_candidate_crop
from pipeline.char_limits import resolve_char_limits, suggested_min_width
from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_json, write_json, write_jsonl
from pipeline.logging_setup import setup_logging
from pipeline.profiles import get_active_profile, allow_ascii_digits
from pipeline.word_limits import resolve_word_limits, suggested_max_crop_height

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hard text-region candidates")
    parser.add_argument("--top-k", type=int, default=50, help="Max candidates to crop")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-per-source", type=int, default=20)
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Minimum words per crop (default: config min_word_count, usually 2)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Maximum words per crop (default: config max_word_count=50; set 100 for longer text)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=None,
        help="Minimum NFC character length (inclusive). When set with --max-chars, char band is enforced.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Maximum NFC character length (inclusive). Enables word-span cropping for short bands.",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    profile = get_active_profile(cfg)
    allow_ascii = allow_ascii_digits(cfg)
    # Dual-lane: pull both hard and ordinary prose; selection enforces quotas later
    min_score = float(args.min_score) if args.min_score else 0.0
    min_words, max_words = resolve_word_limits(cfg, min_words=args.min_words, max_words=args.max_words)
    min_chars, max_chars = resolve_char_limits(cfg, min_chars=args.min_chars, max_chars=args.max_chars)
    if max_chars is not None and args.min_words is None:
        min_words = 1
    logger.info("Word-length filter: %d–%d words (dual-lane extract)", min_words, max_words)
    if min_chars is not None or max_chars is not None:
        logger.info("Character-length filter: %s–%s NFC chars (binding when set)", min_chars, max_chars)
    sources = resolve_path(cfg, "sources")
    candidates_dir = resolve_path(cfg, "candidates")
    crops_dir = candidates_dir / "images"
    crops_dir.mkdir(parents=True, exist_ok=True)

    max_h = suggested_max_crop_height(max_words, int(cfg.get("max_crop_height", 220)))
    min_w = suggested_min_width(max_chars, int(cfg.get("min_width", 800)))

    manifest = read_json(sources / "manifest.json")
    all_cands = []
    per_source: dict[str, int] = {}
    for src in manifest.get("sources", []):
        if Path(src["source_path"]).suffix.lower() != ".pdf":
            continue
        if not Path(src["source_path"]).is_file():
            logger.warning("Missing source PDF: %s", src["source_path"])
            continue
        cands = extract_candidates_from_pdf(
            src["source_path"],
            source_id=src["source_id"],
            source_url=src.get("source_url", ""),
            document_type=src.get("document_type", "unknown"),
            source_document=src.get("filename") or Path(src["source_path"]).name,
            min_complexity=min_score,
            min_words=min_words,
            max_words=max_words,
            min_chars=min_chars,
            max_chars=max_chars,
            allow_ascii_digits=allow_ascii,
            dual_lane=True,
        )
        for c in cands:
            sid = c["source_id"]
            if per_source.get(sid, 0) >= args.max_per_source:
                continue
            per_source[sid] = per_source.get(sid, 0) + 1
            all_cands.append(c)

    all_cands.sort(key=lambda c: c["complexity_score"], reverse=True)
    selected = all_cands[: args.top_k]

    dpi = int(cfg.get("hard_crop_dpi", 600))
    # Tighter margins for short character crops
    margin_x = 12 if (max_chars is not None and max_chars <= 30) else 28
    margin_y = 6 if (max_chars is not None and max_chars <= 30) else 8
    records = []
    seen_text: set[str] = set()
    for i, c in enumerate(selected, 1):
        key = c.get("text_assist", "").strip()
        if key in seen_text:
            continue
        seen_text.add(key)
        out = crops_dir / f"{i:04d}_{c['id']}.png"
        meta = render_candidate_crop(
            c["source_path"],
            c["source_page"],
            c["bbox_discovery_180dpi"],
            out,
            dpi=dpi,
            margin_x=margin_x,
            margin_y=margin_y,
        )
        if meta["image_height"] > max_h:
            logger.info("Skip tall crop %s h=%s (limit=%s)", out.name, meta["image_height"], max_h)
            out.unlink(missing_ok=True)
            continue
        if meta["image_width"] < min_w:
            out.unlink(missing_ok=True)
            continue
        rec = {
            **c,
            **meta,
            "image_filename": str(out.relative_to(ROOT)).replace("\\", "/"),
            "review_status": "pending",
            "ocr_prediction": "",
            "expected_text": "",
            "word_limit": {"min": min_words, "max": max_words},
            "char_limit": {"min": min_chars, "max": max_chars},
        }
        records.append(rec)

    write_jsonl(candidates_dir / "candidates.jsonl", records)
    write_json(
        candidates_dir / "candidates_meta.json",
        {
            "count": len(records),
            "dpi": dpi,
            "min_words": min_words,
            "max_words": max_words,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_width_used": min_w,
        },
    )
    band = f"{min_words}–{max_words} words"
    if min_chars is not None or max_chars is not None:
        band += f", {min_chars}–{max_chars} chars"
    print(f"Wrote {len(records)} candidate crops ({band}) -> {candidates_dir}")


if __name__ == "__main__":
    main()
