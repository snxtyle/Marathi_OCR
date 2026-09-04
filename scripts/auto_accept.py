"""Auto-accept QA-passed candidates — no Streamlit / no human UI.

Label priority:
  1. PDF text_assist (document text layer)
  2. OCR prediction (scanned / visual crops)

Optional: drop samples flagged needs_llm_attention when --strict-vision.
"""

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
from pipeline.splits import classify_issue_type

logger = logging.getLogger(__name__)


def auto_label(rec: dict) -> tuple[str, str]:
    """Return (expected_text, gt_source)."""
    assist = (rec.get("text_assist") or "").strip()
    if assist:
        return assist, "auto_pdf_text"
    ocr = (rec.get("ocr_prediction") or "").strip()
    if ocr:
        return ocr, "auto_ocr"
    expected = (rec.get("expected_text") or "").strip()
    if expected:
        return expected, "auto_existing"
    return "", "auto_empty"


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-accept candidates (no human UI)")
    parser.add_argument(
        "--strict-vision",
        action="store_true",
        help="Reject samples flagged needs_llm_attention / vision mismatch",
    )
    parser.add_argument(
        "--require-pdf-text",
        action="store_true",
        help="Only accept crops that have PDF text_assist (skip OCR-only labels)",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    cand_dir = resolve_path(cfg, "candidates")
    reviewed_dir = resolve_path(cfg, "reviewed")
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    path = cand_dir / "candidates_valid.jsonl"
    if not path.is_file():
        path = cand_dir / "candidates.jsonl"
    records = read_jsonl(path)

    accepted: list[dict] = []
    rejected: list[dict] = []
    for rec in records:
        if args.strict_vision and rec.get("review_status") == "needs_llm_attention":
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["vision_mismatch"]
            rejected.append(rec)
            continue

        text, source = auto_label(rec)
        if not text:
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["empty_auto_label"]
            rejected.append(rec)
            continue
        if args.require_pdf_text and source != "auto_pdf_text":
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["no_pdf_text"]
            rejected.append(rec)
            continue

        rec["expected_text"] = text
        rec["gt_source"] = source
        rec["issue_type"] = rec.get("issue_type") or classify_issue_type(
            text, rec.get("crop_type", "")
        )
        rec["review_status"] = "verified"
        rec["auto_accepted"] = True
        accepted.append(rec)

    write_jsonl(reviewed_dir / "reviewed.jsonl", accepted)
    if rejected:
        write_jsonl(reviewed_dir / "rejected.jsonl", rejected)
    write_json(
        reviewed_dir / "auto_accept_report.json",
        {
            "mode": "auto",
            "accepted": len(accepted),
            "rejected": len(rejected),
            "strict_vision": bool(args.strict_vision),
            "require_pdf_text": bool(args.require_pdf_text),
            "by_gt_source": {
                "auto_pdf_text": sum(1 for r in accepted if r.get("gt_source") == "auto_pdf_text"),
                "auto_ocr": sum(1 for r in accepted if r.get("gt_source") == "auto_ocr"),
                "auto_existing": sum(1 for r in accepted if r.get("gt_source") == "auto_existing"),
            },
            "hard": sum(1 for r in accepted if r.get("difficulty") == "hard"),
            "normal": sum(1 for r in accepted if r.get("difficulty") == "normal"),
        },
    )
    print(
        f"Auto-accepted {len(accepted)} (rejected {len(rejected)}) "
        f"→ {reviewed_dir / 'reviewed.jsonl'}"
    )
    if not accepted:
        sys.exit(2)


if __name__ == "__main__":
    main()
