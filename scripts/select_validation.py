"""Quality-first, source-diverse selection for 80 hard + 20 normal.

Order: quality threshold → source diversity → difficulty → quota.
Never pad shortfalls with weak samples — report e.g. 73/80.

Hard vs normal uses generic heuristic provisional labels (digits / punctuation
density / conjuncts) — never GR keyword lists. Final lane confirmation is the
vision LLM end-gate in validate + auto_accept.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_jsonl, write_json, write_jsonl
from pipeline.logging_setup import setup_logging
from pipeline.profiles import get_active_profile
from scoring.complexity import classify_difficulty, is_ordinary_prose
from validation.duplicates import candidate_dedup_before_ocr, drop_near_duplicate_texts

logger = logging.getLogger(__name__)


def _select_lane(
    records: list[dict],
    *,
    target: int,
    max_per_source: int,
    min_score: float | None = None,
    require_ordinary: bool = False,
    per_source: dict[str, int] | None = None,
) -> tuple[list[dict], dict]:
    """Greedy by score within diversity caps; stop early on shortfall."""
    counts: dict[str, int] = defaultdict(int)
    if per_source:
        for k, v in per_source.items():
            counts[str(k)] = int(v)
    selected: list[dict] = []
    skipped_quality = 0
    skipped_diversity = 0

    if require_ordinary:
        pool = sorted(
            records,
            key=lambda r: (r.get("word_count") or 0, r.get("complexity_score") or 0),
            reverse=True,
        )
    else:
        pool = sorted(records, key=lambda r: r.get("complexity_score") or 0, reverse=True)

    for rec in pool:
        if len(selected) >= target:
            break
        text = rec.get("text_assist") or rec.get("expected_text") or rec.get("ocr_prediction") or ""
        score = float(rec.get("complexity_score") or 0)
        if min_score is not None and score < min_score:
            skipped_quality += 1
            continue
        if require_ordinary and not is_ordinary_prose(text):
            skipped_quality += 1
            continue
        sid = str(rec.get("source_id") or "unknown")
        if counts[sid] >= max_per_source:
            skipped_diversity += 1
            continue
        counts[sid] += 1
        selected.append(rec)

    stats = {
        "target": target,
        "selected": len(selected),
        "shortfall": max(0, target - len(selected)),
        "sources_used": len({str(r.get("source_id")) for r in selected}),
        "per_source": {k: v for k, v in counts.items() if any(str(r.get("source_id")) == k for r in selected)},
        "skipped_quality": skipped_quality,
        "skipped_diversity": skipped_diversity,
        "global_per_source": dict(counts),
    }
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Select source-diverse hard + normal lanes")
    parser.add_argument("--input", default="", help="JSONL input (default candidates_scored/candidates)")
    parser.add_argument("--hard-count", type=int, default=None, help="Override hard lane target")
    parser.add_argument("--normal-count", type=int, default=None, help="Override normal lane target")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg)
    profile = get_active_profile(cfg)
    from pipeline.quotas import scale_lane_targets

    hard_target, normal_target = scale_lane_targets(
        hard_count=args.hard_count,
        normal_count=args.normal_count,
        profile=profile,
        cfg=cfg,
    )
    max_per = int(profile.get("max_samples_per_source", 5))
    min_hard = float(cfg.get("min_complexity_score", 3.0))

    cand_dir = resolve_path(cfg, "candidates")
    in_path = Path(args.input) if args.input else cand_dir / "candidates_scored.jsonl"
    if not in_path.is_file():
        in_path = cand_dir / "candidates.jsonl"
    records = read_jsonl(in_path)

    allow_ascii = bool(profile.get("allow_ascii_digits", False))
    normal_min_words = int(cfg.get("normal_min_words", 6))
    normal_max_words = int(cfg.get("normal_max_words", 40))
    labeled = []
    for rec in records:
        text = rec.get("text_assist") or rec.get("expected_text") or rec.get("ocr_prediction") or ""
        c = classify_difficulty(
            text,
            allow_ascii_digits=allow_ascii,
            min_hard_score=min_hard,
            normal_min_words=normal_min_words,
            normal_max_words=normal_max_words,
        )
        rec["difficulty"] = c["difficulty"]
        rec["complexity_score"] = c["complexity_score"]
        rec["features"] = c.get("features", rec.get("features"))
        rec["hard_signal_source"] = c.get("hard_signal_source")
        rec["contains_marathi_numerals"] = c.get("contains_marathi_numerals")
        rec["contains_reference_identifier"] = c.get("contains_reference_identifier")
        if rec.get("difficulty") in {"hard", "normal"}:
            labeled.append(rec)

    labeled, dedup_stats = candidate_dedup_before_ocr(
        labeled,
        phash_max_distance=int(cfg.get("perceptual_hash_max_distance", 8)),
    )
    # Text near-dup / prefix twins before selection
    labeled, text_dedup_stats = drop_near_duplicate_texts(
        labeled,
        text_key="text_assist",
        threshold=float(cfg.get("text_near_duplicate_threshold", 0.92)),
        prefer_key="complexity_score",
    )

    hard_pool = [r for r in labeled if r.get("difficulty") == "hard"]
    hard_pool = sorted(
        hard_pool,
        key=lambda r: (
            1 if r.get("contains_marathi_numerals") else 0,
            float((r.get("features") or {}).get("identifier_likeness_score") or 0),
            float(r.get("complexity_score") or 0),
        ),
        reverse=True,
    )
    normal_pool = [r for r in labeled if r.get("difficulty") == "normal"]

    hard_sel, hard_stats = _select_lane(
        hard_pool, target=hard_target, max_per_source=max_per, min_score=min_hard
    )
    hard_sources = {r.get("source_id") for r in hard_sel}
    normal_sorted = sorted(
        normal_pool,
        key=lambda r: (0 if r.get("source_id") not in hard_sources else 1, -(r.get("word_count") or 0)),
    )
    normal_sel, normal_stats = _select_lane(
        normal_sorted,
        target=normal_target,
        max_per_source=max_per,
        min_score=None,
        require_ordinary=True,
        per_source=hard_stats.get("global_per_source") or {},
    )

    selected = hard_sel + normal_sel
    write_jsonl(cand_dir / "candidates_selected.jsonl", selected)
    write_jsonl(cand_dir / "candidates.jsonl", selected)
    report = {
        "hard": hard_stats,
        "normal": normal_stats,
        "dedup_before_ocr": dedup_stats,
        "text_near_dedup": text_dedup_stats,
        "quota_met": hard_stats["selected"] >= hard_target and normal_stats["selected"] >= normal_target,
        "note": "shortfall is never padded with weak samples; hard is heuristic-provisional until LLM end-gate",
    }
    write_json(cand_dir / "selection_report.json", report)
    print(
        f"Selected hard={hard_stats['selected']}/{hard_target} "
        f"normal={normal_stats['selected']}/{normal_target} "
        f"sources_hard={hard_stats['sources_used']} "
        f"quota_met={report['quota_met']}"
    )
    if not report["quota_met"]:
        print(
            f"SHORTFALL: hard {hard_stats['selected']}/{hard_target}, "
            f"normal {normal_stats['selected']}/{normal_target} — mine more sources or review inventory."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
