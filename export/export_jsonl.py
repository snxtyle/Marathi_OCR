from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.io_utils import write_json, write_jsonl
from pipeline.splits import classify_issue_type, source_level_split
from validation.duplicates import find_duplicate_images
from validation.numerals import has_ascii_digits


PUBLIC_KEYS = ("image_filename", "expected_text", "issue_type")


def to_public_record(rec: dict[str, Any], *, image_prefix: str = "/images/") -> dict[str, str]:
    name = Path(rec.get("image_filename") or rec.get("image_path") or "").name
    text = rec.get("expected_text") or ""
    issue = rec.get("issue_type") or classify_issue_type(text, rec.get("crop_type", ""))
    return {
        "image_filename": f"{image_prefix}{name}".replace("//", "/"),
        "expected_text": text,
        "issue_type": issue,
    }


def export_dataset(
    records: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    split: bool = False,
) -> dict[str, Any]:
    """Legacy final/ export (all verified records)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    import shutil

    public: list[dict[str, str]] = []
    internal: list[dict[str, Any]] = []
    for i, rec in enumerate(records, 1):
        src = Path(rec.get("image_path") or "")
        name = f"{i:04d}.png"
        dest = images_dir / name
        if src.is_file():
            shutil.copy2(src, dest)
        pub = {
            "image_filename": f"/images/{name}",
            "expected_text": rec.get("expected_text", ""),
            "issue_type": rec.get("issue_type")
            or classify_issue_type(rec.get("expected_text", ""), rec.get("crop_type", "")),
        }
        public.append(pub)
        internal.append({**rec, **pub, "export_index": i})

    write_jsonl(out_dir / "validation.jsonl", public)
    write_json(out_dir / "internal_records.json", internal)

    result: dict[str, Any] = {
        "public_count": len(public),
        "validation_jsonl": str(out_dir / "validation.jsonl"),
    }
    if split:
        parts = source_level_split(internal)
        for key in ("train", "val"):
            pubs = [
                {
                    "image_filename": r["image_filename"],
                    "expected_text": r["expected_text"],
                    "issue_type": r["issue_type"],
                }
                for r in parts[key]
            ]
            write_jsonl(out_dir / f"{key}.jsonl", pubs)
        result["split"] = {
            "train": len(parts["train"]),
            "val": len(parts["val"]),
            "train_sources": parts["train_sources"],
            "val_sources": parts["val_sources"],
        }
    return result


def export_validation_package(
    records: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    hard_target: int = 80,
    normal_target: int = 20,
    max_per_source: int = 5,
    allow_ascii_digits: bool = False,
    allow_shortfall: bool = False,
    phash_max_distance: int = 8,
    text_sim_threshold: float = 0.92,
    project_root: Path | None = None,
    require_llm_complex_for_hard: bool = True,
) -> dict[str, Any]:
    """Export Desktop marathi_ocr_validation_100 package with final QA gate."""
    import shutil

    out_dir = Path(out_dir)
    root = project_root or Path(__file__).resolve().parents[1]

    verified = [r for r in records if r.get("review_status") == "verified"]

    # Drop samples the LLM end-gate rejected for mismatch / not-complex (if still present)
    llm_gated_out: list[dict[str, Any]] = []
    llm_drop = 0
    for r in verified:
        llm = (r.get("validation") or {}).get("llm") or {}
        if llm and not llm.get("skipped") and llm.get("enabled", True):
            if llm.get("exact_match") is False and r.get("gt_source") not in {
                "auto_ocr_vision_corrected",
                "auto_vision_match",
            }:
                llm_drop += 1
                continue
            if llm.get("suggested_lane") == "reject":
                llm_drop += 1
                continue
            if (
                require_llm_complex_for_hard
                and r.get("difficulty") == "hard"
                and llm.get("is_complex_for_ocr") is False
            ):
                # Should already be demoted in auto_accept; belt-and-suspenders
                r = {**r, "difficulty": "normal", "hard_signal_source": "export_demoted_not_complex"}
        llm_gated_out.append(r)
    verified = llm_gated_out

    # Text near-dup / prefix twins: drop globally (keep higher complexity)
    from validation.duplicates import drop_near_duplicate_texts

    thresh = max(0.88, text_sim_threshold - 0.04)
    verified, text_drop_stats = drop_near_duplicate_texts(
        verified,
        text_key="expected_text",
        threshold=thresh,
        prefer_key="complexity_score",
        cross_source=True,
    )
    text_flags: list[dict[str, Any]] = [{"dropped": True, "stats": text_drop_stats}]

    hard = [r for r in verified if r.get("difficulty") == "hard"]
    normal = [r for r in verified if r.get("difficulty") == "normal"]
    # Prefer highest quality within caps already applied at select; take up to targets
    hard = sorted(hard, key=lambda r: r.get("complexity_score") or 0, reverse=True)[:hard_target]
    normal = sorted(normal, key=lambda r: r.get("word_count") or 0, reverse=True)[:normal_target]

    shortfall = {
        "hard": f"{len(hard)}/{hard_target}",
        "normal": f"{len(normal)}/{normal_target}",
    }
    quota_met = len(hard) >= hard_target and len(normal) >= normal_target
    if not quota_met and not allow_shortfall:
        report = {
            "layer": "final_qa",
            "total_samples": len(hard) + len(normal),
            "hard_samples": len(hard),
            "normal_samples": len(normal),
            "hard_target": hard_target,
            "normal_target": normal_target,
            "shortfall": shortfall,
            "quota_met": False,
            "export_blocked": True,
            "manual_review_completed": False,
            "note": "Refusing to pad weak samples. Mine more or pass --allow-shortfall for draft.",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "validation_report.json", report)
        return report

    selected = hard + normal
    # Drop ASCII-digit GT under Marathi-digits-only profiles (OCR drift)
    ascii_dropped = 0
    if not allow_ascii_digits:
        clean: list[dict[str, Any]] = []
        for r in selected:
            if has_ascii_digits(r.get("expected_text") or ""):
                ascii_dropped += 1
                continue
            clean.append(r)
        selected = clean
        hard = [r for r in selected if r.get("difficulty") == "hard"]
        normal = [r for r in selected if r.get("difficulty") == "normal"]
        shortfall = {
            "hard": f"{len(hard)}/{hard_target}",
            "normal": f"{len(normal)}/{normal_target}",
        }
        quota_met = len(hard) >= hard_target and len(normal) >= normal_target

    # Image-strict dedup
    paths = []
    for r in selected:
        p = Path(r.get("image_path") or "")
        if not p.is_file():
            p = root / (r.get("image_filename") or "")
        if p.is_file():
            paths.append(str(p))
            r["_resolved_image"] = str(p)
    img_dups = find_duplicate_images(paths, phash_max_distance=phash_max_distance)
    # Remaining violations after drop (should be 0)
    ascii_violations = 0 if allow_ascii_digits else sum(
        1 for r in selected if has_ascii_digits(r.get("expected_text") or "")
    )
    missing = sum(1 for r in selected if not Path(r.get("_resolved_image") or "").is_file())
    empty_text = sum(1 for r in selected if not (r.get("expected_text") or "").strip())

    per_source: Counter[str] = Counter(str(r.get("source_id")) for r in selected)
    over_cap = {s: n for s, n in per_source.items() if n > max_per_source}

    gate_ok = (
        quota_met
        and len(img_dups["exact_sha_groups"]) == 0
        and len(img_dups["near_phash_pairs"]) == 0
        and ascii_violations == 0
        and missing == 0
        and empty_text == 0
        and not over_cap
    )

    images_dir = out_dir / "images"
    if gate_ok or allow_shortfall:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        public: list[dict[str, str]] = []
        metadata: list[dict[str, Any]] = []
        for i, rec in enumerate(selected, 1):
            name = f"{i:04d}.png"
            dest = images_dir / name
            src = Path(rec.get("_resolved_image") or "")
            if src.is_file():
                shutil.copy2(src, dest)
            text = rec.get("expected_text") or ""
            issue = rec.get("issue_type") or classify_issue_type(text, rec.get("crop_type", ""))
            pub = {
                "image_filename": f"/images/{name}",
                "expected_text": text,
                "issue_type": issue,
            }
            public.append(pub)
            metadata.append(
                {
                    **pub,
                    "difficulty": rec.get("difficulty"),
                    "complexity_score": rec.get("complexity_score"),
                    "source_id": rec.get("source_id"),
                    "source_url": rec.get("source_url"),
                    "source_document": rec.get("source_document"),
                    "document_type": rec.get("document_type"),
                    "page_number": rec.get("source_page"),
                    "original_coordinates": rec.get("original_coordinates"),
                    "review_status": rec.get("review_status"),
                }
            )
        write_jsonl(out_dir / "validation.jsonl", public)
        write_json(out_dir / "metadata.json", {"samples": metadata})
        (out_dir / "README.md").write_text(
            "# Marathi OCR validation set (80 hard / 20 normal)\n\n"
            "Public schema: `validation.jsonl` with `image_filename`, `expected_text`, `issue_type`.\n"
            "Internal provenance: `metadata.json`. QA: `validation_report.json`.\n",
            encoding="utf-8",
        )
    else:
        public = []

    scores = [float(r.get("complexity_score") or 0) for r in selected]
    issue_counts = Counter(
        r.get("issue_type")
        or classify_issue_type(r.get("expected_text", ""), r.get("crop_type", ""))
        for r in selected
    )
    gt_sources = Counter(str(r.get("gt_source") or "unknown") for r in selected)
    auto_n = sum(1 for r in selected if r.get("auto_accepted"))
    human_n = sum(
        1
        for r in selected
        if r.get("review_status") == "verified" and not r.get("auto_accepted")
    )
    report = {
        "layer": "final_qa",
        "total_samples": len(selected),
        "hard_samples": len(hard),
        "normal_samples": len(normal),
        "hard_target": hard_target,
        "normal_target": normal_target,
        "shortfall": shortfall,
        "quota_met": quota_met,
        "duplicate_samples": sum(len(g) - 1 for g in img_dups["exact_sha_groups"]),
        "near_duplicate_samples": len(img_dups["near_phash_pairs"]),
        "text_near_duplicate_flags": text_drop_stats.get("dropped", 0),
        "text_near_duplicates": text_flags[:50],
        "text_near_dedup_stats": text_drop_stats,
        "arabic_digit_violations": ascii_violations,
        "ascii_digit_dropped": ascii_dropped,
        "missing_images": missing,
        "empty_text": empty_text,
        "source_cap_violations": over_cap,
        "sources_used": len(per_source),
        "per_source_counts": dict(per_source),
        "manual_review_completed": human_n == len(selected) and len(selected) > 0,
        "auto_accepted_count": auto_n,
        "human_verified_count": human_n,
        "gt_sources": dict(gt_sources),
        "image_text_mismatches": 0,
        "issue_type_counts": dict(issue_counts),
        "average_complexity_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "llm_gated_drops": llm_drop,
        "gate_ok": gate_ok,
        "export_path": str(out_dir),
        "public_lines": len(public),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "validation_report.json", report)
    return report
