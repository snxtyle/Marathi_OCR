"""Stage A candidate dedup (bbox / exact image / phash) before OCR."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_jsonl, write_json, write_jsonl
from pipeline.logging_setup import setup_logging
from validation.duplicates import candidate_dedup_before_ocr

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate dedup before OCR (Stage A)")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    cand_dir = resolve_path(cfg, "candidates")
    in_path = Path(args.input) if args.input else cand_dir / "candidates_scored.jsonl"
    if not in_path.is_file():
        in_path = cand_dir / "candidates.jsonl"
    out_path = Path(args.output) if args.output else cand_dir / "candidates.jsonl"

    records = read_jsonl(in_path)
    kept, stats = candidate_dedup_before_ocr(
        records,
        phash_max_distance=int(cfg.get("perceptual_hash_max_distance", 8)),
    )
    write_jsonl(out_path, kept)
    write_jsonl(cand_dir / "candidates_deduped.jsonl", kept)
    write_json(cand_dir / "dedup_stage_a_report.json", {"layer": "candidate_dedup", **stats})
    print(f"Stage A dedup: {stats['kept']}/{stats['input']} kept ({stats})")


if __name__ == "__main__":
    main()
