from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from export.export_jsonl import export_dataset, export_validation_package
from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_jsonl
from pipeline.logging_setup import setup_logging
from pipeline.profiles import get_active_profile

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export verified samples (final/ or validation package)")
    parser.add_argument("--split", action="store_true", help="Also write train/val by source_id (legacy final/)")
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Include pending OCR pre-labels (MVP bootstrap only)",
    )
    parser.add_argument(
        "--validation-package",
        action="store_true",
        help="Export 80/20 package to validation_export_dir with final QA gate",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow draft export when hard/normal quotas are incomplete (never pads quality)",
    )
    parser.add_argument("--out", default="", help="Override output directory")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    profile = get_active_profile(cfg)
    reviewed_dir = resolve_path(cfg, "reviewed")
    cand_dir = resolve_path(cfg, "candidates")
    final_dir = resolve_path(cfg, "final")

    records = read_jsonl(reviewed_dir / "reviewed.jsonl")
    if not records and args.include_pending:
        records = [
            r
            for r in read_jsonl(cand_dir / "candidates.jsonl")
            if (r.get("expected_text") or r.get("ocr_prediction"))
        ]
        for r in records:
            r["expected_text"] = r.get("expected_text") or r.get("ocr_prediction")
            r["review_status"] = r.get("review_status") or "pending_export"

    if args.validation_package:
        if args.out:
            out = Path(args.out)
        else:
            raw_out = cfg.get("validation_export_dir") or "data/export/marathi_ocr_validation_100"
            out = Path(raw_out)
            if not out.is_absolute():
                out = ROOT / out
        report = export_validation_package(
            records,
            out,
            hard_target=int(profile.get("hard_count", 80)),
            normal_target=int(profile.get("normal_count", 20)),
            max_per_source=int(profile.get("max_samples_per_source", 5)),
            allow_ascii_digits=bool(profile.get("allow_ascii_digits", False)),
            allow_shortfall=bool(args.allow_shortfall),
            phash_max_distance=int(cfg.get("perceptual_hash_max_distance", 8)),
            text_sim_threshold=float(cfg.get("text_near_duplicate_threshold", 0.92)),
            project_root=ROOT,
        )
        print(report)
        if report.get("export_blocked"):
            sys.exit(2)
        return

    verified = [
        r
        for r in records
        if r.get("review_status") in {"verified", "pending_export"} or args.include_pending
    ]
    if not verified:
        print("No verified records to export. Run review UI or pass --include-pending.")
        return

    result = export_dataset(verified, final_dir if not args.out else Path(args.out), split=args.split)
    print(result)


if __name__ == "__main__":
    main()
