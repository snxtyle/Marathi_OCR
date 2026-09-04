from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring.complexity import classify_difficulty, visual_complexity_score
from pipeline.char_limits import resolve_char_limits
from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_jsonl, write_jsonl
from pipeline.logging_setup import setup_logging
from pipeline.profiles import allow_ascii_digits, get_active_profile
from pipeline.word_limits import resolve_word_limits

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="(Re)score candidates; assign hard|normal|reject")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg)
    profile = get_active_profile(cfg)
    allow_ascii = allow_ascii_digits(cfg)
    min_hard = float(args.min_score or cfg.get("min_complexity_score", 3.0))
    max_normal = float(cfg.get("normal_max_complexity_score", 2.5))
    min_words, max_words = resolve_word_limits(cfg, min_words=args.min_words, max_words=args.max_words)
    min_chars, max_chars = resolve_char_limits(cfg, min_chars=args.min_chars, max_chars=args.max_chars)
    mode = str(cfg.get("scoring_mode") or "auto").lower()
    cand_dir = resolve_path(cfg, "candidates")
    records = read_jsonl(cand_dir / "candidates.jsonl")
    kept = []
    for rec in records:
        text = rec.get("text_assist") or rec.get("ocr_prediction") or rec.get("expected_text") or ""
        score_override = None
        use_visual = mode == "visual" or (mode == "auto" and not text.strip())
        if use_visual:
            score_override = visual_complexity_score(
                width=int(rec.get("image_width") or 0),
                height=int(rec.get("image_height") or 0),
                aspect_ratio=(
                    float(rec.get("image_width") or 0) / max(float(rec.get("image_height") or 1), 1.0)
                ),
                cheap_ocr_text=rec.get("ocr_prediction") or "",
            )
            # Preserve visual-band candidates without PDF text for human GT
            if not text.strip():
                rec.update(
                    {
                        "complexity_score": score_override,
                        "difficulty": "hard" if score_override >= min_hard else "normal",
                        "scoring_mode_used": "visual",
                        "word_count": rec.get("word_count") or 0,
                        "char_count": rec.get("char_count") or 0,
                        "features": rec.get("features") or {"visual_score": score_override},
                        "contains_marathi_numerals": False,
                        "contains_special_chars": False,
                        "contains_reference_identifier": False,
                        "contains_conjuncts": False,
                    }
                )
                if rec["difficulty"] in {"hard", "normal"}:
                    kept.append(rec)
                continue
        classified = classify_difficulty(
            text,
            min_hard_score=min_hard,
            max_normal_score=max_normal,
            min_words=min_words,
            max_words=max_words,
            normal_min_words=int(cfg.get("normal_min_words", 6)),
            normal_max_words=int(cfg.get("normal_max_words", 40)),
            min_chars=min_chars,
            max_chars=max_chars,
            allow_ascii_digits=allow_ascii,
            score_override=score_override,
        )
        rec.update(
            {
                "complexity_score": classified["complexity_score"],
                "features": classified["features"],
                "word_count": classified["word_count"],
                "char_count": classified["char_count"],
                "difficulty": classified["difficulty"],
                "scoring_mode_used": "visual" if use_visual else "text",
                "word_limit": {"min": min_words, "max": max_words},
                "char_limit": {"min": min_chars, "max": max_chars},
                "contains_marathi_numerals": classified["contains_marathi_numerals"],
                "contains_special_chars": classified["contains_special_chars"],
                "contains_reference_identifier": classified["contains_reference_identifier"],
                "contains_conjuncts": classified["contains_conjuncts"],
            }
        )
        if classified["difficulty"] in {"hard", "normal"}:
            kept.append(rec)
    write_jsonl(cand_dir / "candidates.jsonl", kept)
    write_jsonl(cand_dir / "candidates_scored.jsonl", kept)
    n_hard = sum(1 for r in kept if r.get("difficulty") == "hard")
    n_normal = sum(1 for r in kept if r.get("difficulty") == "normal")
    print(
        f"Scored/kept {len(kept)}/{len(records)} "
        f"(hard={n_hard} normal={n_normal} profile={profile.get('_name')})"
    )


if __name__ == "__main__":
    main()
