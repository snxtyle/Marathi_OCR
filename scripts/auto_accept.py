"""Auto-accept QA-passed candidates — no Streamlit / no human UI.

Vision-aware labeling (default):
  - If Kimi vision says image↔text MATCH → keep that text
  - If MISMATCH → prefer OCR (PDF text layers are often garbled); reject if no OCR
  - If no vision result → PDF text_assist, else OCR
  - Apply LLM ocr_complexity / suggested_lane (hard|normal|reject)
  - Soft-fail: demote unconfirmed hard (never fill hard with title-slash prose)
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
from pipeline.profiles import get_active_profile
from pipeline.splits import classify_issue_type
from scoring.complexity import apply_llm_complexity, demote_unconfirmed_hard
from validation.numerals import has_ascii_digits

logger = logging.getLogger(__name__)


def _llm_block(rec: dict) -> dict:
    return (rec.get("validation") or {}).get("llm") or {}


def choose_label(rec: dict, *, trust_vision: bool = True) -> tuple[str, str, str | None]:
    """Return (expected_text, gt_source, reject_reason|None)."""
    assist = (rec.get("text_assist") or "").strip()
    ocr = (rec.get("ocr_prediction") or "").strip()
    existing = (rec.get("expected_text") or "").strip()
    llm = _llm_block(rec)

    vision_ran = bool(llm) and not llm.get("skipped") and llm.get("enabled", True)
    exact = llm.get("exact_match")

    if trust_vision and vision_ran and exact is True:
        text = assist or existing or ocr
        if text:
            return text, "auto_vision_match", None
        return "", "auto_empty", "empty_after_vision_match"

    if trust_vision and vision_ran and exact is False:
        if ocr and ocr != assist:
            return ocr, "auto_ocr_vision_corrected", None
        if ocr:
            return ocr, "auto_ocr_vision_corrected", None
        return "", "auto_empty", "vision_mismatch_no_ocr"

    if assist:
        return assist, "auto_pdf_text", None
    if ocr:
        return ocr, "auto_ocr", None
    if existing:
        return existing, "auto_existing", None
    return "", "auto_empty", "empty_auto_label"


def _apply_complexity_gate(
    rec: dict,
    *,
    allow_ascii: bool,
    min_hard: float,
    trust_vision: bool,
) -> str | None:
    """Return reject reason or None. Mutates rec difficulty from LLM lane."""
    llm = _llm_block(rec)
    skipped = bool(llm.get("skipped")) or not llm.get("enabled", True)
    lane = (
        llm.get("ocr_complexity")
        or llm.get("suggested_lane")
        or rec.get("llm_suggested_lane")
        or ""
    )
    lane = str(lane).strip().lower()
    if not lane and not skipped:
        # Derive from boolean when model omitted suggested_lane
        if llm.get("is_complex_for_ocr") is True:
            lane = "hard"
        elif llm.get("is_complex_for_ocr") is False:
            # Heuristic hard + LLM says not complex → normal (not reject)
            lane = "normal" if rec.get("difficulty") == "hard" else "normal"

    if trust_vision and not skipped and lane in {"hard", "normal", "reject"}:
        apply_llm_complexity(
            rec,
            lane,
            allow_ascii_digits=allow_ascii,
            min_hard_score=min_hard,
        )
        if rec.get("difficulty") == "reject":
            return "llm_complexity_reject"
        return None

    # Soft-fail / ignore-vision: never keep weak hard without confirmation
    demote_unconfirmed_hard(rec)
    if rec.get("difficulty") == "reject":
        return "hard_unconfirmed_llm_soft_fail"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-accept candidates (no human UI)")
    parser.add_argument(
        "--strict-vision",
        action="store_true",
        help="Reject when vision mismatch cannot be corrected via OCR",
    )
    parser.add_argument(
        "--ignore-vision",
        action="store_true",
        help="Do not use Kimi vision results when choosing labels (not recommended)",
    )
    parser.add_argument(
        "--require-pdf-text",
        action="store_true",
        help="Only accept crops that keep PDF text_assist (skip OCR-corrected labels)",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    profile = get_active_profile(cfg)
    allow_ascii = bool(profile.get("allow_ascii_digits", False))
    min_hard = float(cfg.get("min_complexity_score", 3.0))
    cand_dir = resolve_path(cfg, "candidates")
    reviewed_dir = resolve_path(cfg, "reviewed")
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    path = cand_dir / "candidates_valid.jsonl"
    if not path.is_file():
        path = cand_dir / "candidates.jsonl"
    records = read_jsonl(path)

    accepted: list[dict] = []
    rejected: list[dict] = []
    corrected = 0
    complexity_rejects = 0

    for rec in records:
        text, source, reject_reason = choose_label(rec, trust_vision=not args.ignore_vision)

        if reject_reason:
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + [reject_reason]
            rejected.append(rec)
            continue

        if not text:
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["empty_auto_label"]
            rejected.append(rec)
            continue

        if not allow_ascii and has_ascii_digits(text):
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["ascii_digits"]
            rejected.append(rec)
            continue

        if args.require_pdf_text and source not in {"auto_pdf_text", "auto_vision_match"}:
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["no_pdf_text"]
            rejected.append(rec)
            continue

        cx_reason = _apply_complexity_gate(
            rec,
            allow_ascii=allow_ascii,
            min_hard=min_hard,
            trust_vision=not args.ignore_vision,
        )
        if cx_reason:
            complexity_rejects += 1
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + [cx_reason]
            rejected.append(rec)
            continue

        if rec.get("difficulty") not in {"hard", "normal"}:
            rec["review_status"] = "rejected"
            rec["reject_reasons"] = rec.get("reject_reasons", []) + ["no_lane"]
            rejected.append(rec)
            continue

        if source == "auto_ocr_vision_corrected":
            corrected += 1

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
            "mode": "auto_vision_complexity_aware",
            "accepted": len(accepted),
            "rejected": len(rejected),
            "vision_ocr_corrected": corrected,
            "complexity_rejects": complexity_rejects,
            "strict_vision": bool(args.strict_vision),
            "ignore_vision": bool(args.ignore_vision),
            "require_pdf_text": bool(args.require_pdf_text),
            "by_gt_source": {
                "auto_pdf_text": sum(1 for r in accepted if r.get("gt_source") == "auto_pdf_text"),
                "auto_ocr": sum(1 for r in accepted if r.get("gt_source") == "auto_ocr"),
                "auto_ocr_vision_corrected": sum(
                    1 for r in accepted if r.get("gt_source") == "auto_ocr_vision_corrected"
                ),
                "auto_vision_match": sum(
                    1 for r in accepted if r.get("gt_source") == "auto_vision_match"
                ),
                "auto_existing": sum(1 for r in accepted if r.get("gt_source") == "auto_existing"),
            },
            "hard": sum(1 for r in accepted if r.get("difficulty") == "hard"),
            "normal": sum(1 for r in accepted if r.get("difficulty") == "normal"),
        },
    )
    print(
        f"Auto-accepted {len(accepted)} (rejected {len(rejected)}, "
        f"vision→OCR corrected {corrected}, complexity_rejects {complexity_rejects}) "
        f"→ {reviewed_dir / 'reviewed.jsonl'}"
    )
    if not accepted:
        sys.exit(2)


if __name__ == "__main__":
    main()
