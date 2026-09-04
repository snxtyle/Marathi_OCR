from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from benchmark.metrics import aggregate, score_pair
from ocr.factory import create_backend
from pipeline.cache import OCRCache
from pipeline.config import load_config, resolve_path
from pipeline.hashing import sha256_file
from pipeline.io_utils import read_jsonl, write_json
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def run_backend(name: str, labels: list[dict], bench_dir: Path, cache: OCRCache, use_gpu: bool) -> dict:
    try:
        backend = create_backend(name, use_gpu=use_gpu, lang="mar")
    except Exception as exc:  # noqa: BLE001
        logger.error("Backend %s unavailable: %s", name, exc)
        return {"backend": name, "error": str(exc), "metrics": None, "rows": []}

    rows = []
    for rec in labels:
        img_path = bench_dir / rec["image_filename"]
        if not img_path.is_file():
            continue
        digest = sha256_file(img_path)
        cached = cache.get(digest, backend.name, backend.model_version)
        latency_ms: float | None = None
        from_cache = False
        if cached and "text" in cached:
            hyp = cached["text"]
            from_cache = True
            if isinstance(cached.get("latency_ms"), (int, float)):
                latency_ms = float(cached["latency_ms"])
        else:
            t0 = time.perf_counter()
            with Image.open(img_path) as img:
                hyp = backend.recognize(img)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            cache.set(
                digest,
                backend.name,
                backend.model_version,
                {
                    "text": hyp,
                    "image": str(img_path),
                    "latency_ms": latency_ms,
                    "sha256": digest,
                },
            )
        meta = {
            "issue_type": rec.get("issue_type", ""),
            "source_id": rec.get("source_id", ""),
        }
        metrics = score_pair(rec["expected_text"], hyp, meta=meta)
        rows.append(
            {
                "image": rec["image_filename"],
                "expected": rec["expected_text"],
                "hypothesis": hyp,
                "latency_ms": latency_ms,
                "from_cache": from_cache,
                **metrics,
            }
        )
    summary = aggregate(rows)
    return {
        "backend": name,
        "model_version": backend.model_version,
        "metrics": summary,
        "rows": rows,
    }


def pick_winner(results: list[dict]) -> str:
    scored = []
    for r in results:
        m = r.get("metrics")
        if not m:
            continue
        # Prefer higher EM / numeral / punct F1 / ref-id; lower CER/WER/conjunct errors
        conj = m.get("conjunct_error_rate")
        conj_pen = (conj if conj is not None else 0.0) * 0.5
        contam = m.get("ascii_digit_contamination_rate")
        contam_pen = (contam if contam is not None else 0.0) * 1.5
        score = (
            (m.get("exact_match") or 0) * 3
            + (m.get("marathi_numeral_accuracy") or 0) * 2
            + (m.get("special_char_f1") or m.get("special_char_accuracy") or 0) * 1.5
            + (m.get("reference_id_exact_match") or 0) * 2
            - (m.get("cer_normalized") or m.get("cer") or 0) * 2
            - (m.get("wer") or 0)
            - conj_pen
            - contam_pen
        )
        scored.append((score, r["backend"]))
    if not scored:
        return "tesseract"
    scored.sort(reverse=True)
    return scored[0][1]


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def render_markdown(report: dict) -> str:
    lines = [
        "# Marathi OCR Hard Benchmark Report",
        "",
        f"Samples: {report['n_labels']}",
        f"Winner: **{report['winner']}**",
        f"Generated: {report.get('generated_at', '')}",
        "",
        "## Metric definitions (summary)",
        "",
        "- **exact_match**: full-string equality after NFC + strip ZWJ/ZWNJ + collapse whitespace",
        "- **cer / cer_normalized**: Levenshtein / len(GT) on normalized strings",
        "- **cer_strict**: same on as-is (outer-stripped) strings",
        "- **wer**: token-level Levenshtein / len(GT tokens)",
        "- **marathi_numeral_accuracy**: subset CER on Devanagari digits ०-९",
        "- **ascii_digit_contamination_rate**: share of Marathi-digit GT samples whose pred contains ASCII 0-9",
        "- **reference_id_exact_match**: exact_match on reference-like GT subset",
        "- **special_char_f1**: multiset F1 over `/ . , - () : ;` etc.",
        "- **conjunct_error_rate**: missing named/virama clusters (क्ष त्र ज्ञ श्र द्व र् …) vs GT",
        "- **length_band_cer**: short (≤25 chars) vs longer; plus word-count bands",
        "- **latency_ms_p50/p95**: recognition time per crop (live calls; cached latency if stored)",
        "",
    ]
    for r in report["results"]:
        lines.append(f"## {r['backend']}")
        if r.get("error"):
            lines.append(f"- Error: `{r['error']}`")
            lines.append("")
            continue
        m = r.get("metrics") or {}
        lines.append(f"- Model: `{r.get('model_version')}`")
        lines.append(f"- CER (normalized): {_fmt(m.get('cer_normalized') or m.get('cer'))}")
        lines.append(f"- CER (strict): {_fmt(m.get('cer_strict'))}")
        lines.append(f"- WER: {_fmt(m.get('wer'))}")
        lines.append(f"- Exact Match: {_fmt(m.get('exact_match'))}")
        lines.append(f"- Exact Match (strict): {_fmt(m.get('exact_match_strict'))}")
        lines.append(f"- Marathi Numeral Acc: {_fmt(m.get('marathi_numeral_accuracy'))}")
        lines.append(f"- ASCII Digit Contamination: {_fmt(m.get('ascii_digit_contamination_rate'))}")
        lines.append(f"- Special Char Acc: {_fmt(m.get('special_char_accuracy'))}")
        lines.append(f"- Special Char F1: {_fmt(m.get('special_char_f1'))}")
        lines.append(f"- Reference-ID Exact: {_fmt(m.get('reference_id_exact_match'))} (n={m.get('reference_id_applicable', 0)})")
        lines.append(f"- Conjunct Error Rate: {_fmt(m.get('conjunct_error_rate'))} (n={m.get('conjunct_applicable', 0)})")
        lines.append(
            f"- Latency ms p50/p95: {_fmt(m.get('latency_ms_p50'), 1)} / {_fmt(m.get('latency_ms_p95'), 1)} "
            f"(n={m.get('latency_samples', 0)})"
        )
        lb = m.get("length_band_cer") or {}
        if lb:
            lines.append("- Length-band CER:")
            for band, vals in lb.items():
                lines.append(
                    f"  - `{band}` n={vals.get('n')}: "
                    f"norm={_fmt(vals.get('cer_normalized') or vals.get('cer'))} "
                    f"strict={_fmt(vals.get('cer_strict'))}"
                )
        slices = m.get("slices") or {}
        by_issue = slices.get("by_issue_type") or {}
        if by_issue:
            lines.append("- Slice by issue_type (CER / EM):")
            for issue, vals in by_issue.items():
                lines.append(
                    f"  - `{issue or 'unknown'}` n={vals.get('n')}: "
                    f"cer={_fmt(vals.get('cer'))} em={_fmt(vals.get('exact_match'))}"
                )
        by_digits = slices.get("by_contains_marathi_digits") or {}
        if by_digits:
            lines.append("- Slice by contains_marathi_digits:")
            for label, vals in by_digits.items():
                lines.append(
                    f"  - `{label}` n={vals.get('n')}: cer={_fmt(vals.get('cer'))} "
                    f"num_acc={_fmt(vals.get('marathi_numeral_accuracy'))}"
                )
        by_src = slices.get("by_source_id") or {}
        if by_src.get("n_sources"):
            lines.append(
                f"- Source_id aggregates: {by_src['n_sources']} sources"
                + (f" ({by_src.get('omitted_sources', 0)} omitted from table)" if by_src.get("omitted_sources") else "")
            )
            # Show compact top few by n
            items = sorted(
                (by_src.get("sources") or {}).items(),
                key=lambda kv: (-kv[1].get("n", 0), kv[0]),
            )[:8]
            for sid, vals in items:
                lines.append(
                    f"  - `{sid}` n={vals.get('n')}: cer={_fmt(vals.get('cer'))} em={_fmt(vals.get('exact_match'))}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR benchmark on hard Marathi crops")
    parser.add_argument(
        "--backends",
        default="tesseract,paddleocr,surya",
        help="Comma-separated backends to bake off (writes winner to ocr_backend)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Do not rewrite configs/config.yaml ocr_backend to the winner",
    )
    parser.add_argument(
        "--write-rows",
        action="store_true",
        help="Also write per-sample rows JSON under benchmark/runs/<ts>/",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    bench = resolve_path(cfg, "benchmark")
    labels = read_jsonl(bench / "labels.jsonl")
    if args.limit:
        labels = labels[: args.limit]
    cache = OCRCache(bench / "cache" / "ocr")
    use_gpu = bool(cfg.get("ocr_use_gpu", True))

    results = []
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        logger.info("Running backend=%s on %d samples", name, len(labels))
        results.append(run_backend(name, labels, bench, cache, use_gpu))

    winner = pick_winner(results)
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "n_labels": len(labels),
        "winner": winner,
        "generated_at": generated_at,
        "metric_schema": [
            "cer",
            "cer_strict",
            "cer_normalized",
            "wer",
            "exact_match",
            "exact_match_strict",
            "marathi_numeral_accuracy",
            "ascii_digit_contamination_rate",
            "special_char_accuracy",
            "special_char_f1",
            "special_char_precision",
            "special_char_recall",
            "reference_id_exact_match",
            "conjunct_error_rate",
            "length_band_cer",
            "word_band_cer",
            "latency_ms_p50",
            "latency_ms_p95",
            "slices.by_issue_type",
            "slices.by_contains_marathi_digits",
            "slices.by_source_id",
        ],
        "results": [
            {
                "backend": r["backend"],
                "model_version": r.get("model_version"),
                "error": r.get("error"),
                "metrics": r.get("metrics"),
            }
            for r in results
        ],
    }
    write_json(bench / "report.json", report)
    md = render_markdown(report)
    (bench / "REPORT.md").write_text(md, encoding="utf-8")

    run_dir = bench / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "report.json", report)
    (run_dir / "REPORT.md").write_text(md, encoding="utf-8")
    if args.write_rows:
        for r in results:
            safe = re_sub_backend(r["backend"])
            write_json(
                run_dir / f"rows_{safe}.json",
                {"backend": r["backend"], "model_version": r.get("model_version"), "rows": r.get("rows") or []},
            )

    if not args.no_update_config:
        cfg_path = ROOT / "configs" / "config.yaml"
        if cfg_path.is_file():
            text = cfg_path.read_text(encoding="utf-8")
            if "ocr_backend:" in text:
                import re

                text = re.sub(r"^ocr_backend:.*$", f"ocr_backend: {winner}", text, count=1, flags=re.M)
                cfg_path.write_text(text, encoding="utf-8")

    print(f"Winner: {winner}")
    print(f"Wrote {bench / 'report.json'} and REPORT.md")
    print(f"Run snapshot: {run_dir}")


def re_sub_backend(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


if __name__ == "__main__":
    main()
