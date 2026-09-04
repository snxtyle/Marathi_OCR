"""Fully automatic Marathi OCR validation factory — one command, no UI.

discover → render → extract → score → dedup → select → ocr → validate
  → auto_accept → export validation package
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(module: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module, *(extra or [])]
    print(f"\n==> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fully automatic validation factory (no Streamlit / no manual review)"
    )
    parser.add_argument("--max-docs", type=int, default=8)
    parser.add_argument("--skip-discover", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        default=True,
        help="Export whatever quality-gated inventory exists (default on for auto mode)",
    )
    parser.add_argument(
        "--strict-quota",
        action="store_true",
        help="Block export unless hard/normal targets are fully met",
    )
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-per-source", type=int, default=12)
    parser.add_argument(
        "--strict-vision",
        action="store_true",
        help="Drop vision-flagged mismatches during auto-accept",
    )
    parser.add_argument(
        "--require-pdf-text",
        action="store_true",
        help="Only keep crops with PDF text layer labels (no OCR-only GT)",
    )
    parser.add_argument(
        "--with-review",
        action="store_true",
        help="Optional: launch Streamlit human review instead of auto-accept",
    )
    parser.add_argument("--skip-llm-validate", action="store_true", help="Skip Kimi vision in validate")
    args = parser.parse_args()

    word_args: list[str] = []
    if args.max_words is not None:
        word_args = ["--max-words", str(args.max_words)]

    if not args.skip_discover:
        _run("scripts.discover", ["--max-docs", str(args.max_docs)])

    _run("scripts.render")
    _run(
        "scripts.extract",
        ["--top-k", str(args.top_k), "--max-per-source", str(args.max_per_source), *word_args],
    )
    _run("scripts.score", word_args)
    _run("scripts.dedup_candidates")
    try:
        _run("scripts.select_validation")
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 2:
            raise
        print("Selection shortfall — continuing with quality-gated inventory.")

    _run("scripts.ocr")
    validate_args = list(word_args)
    if args.skip_llm_validate:
        validate_args.append("--skip-llm")
    _run("scripts.validate", validate_args)

    if args.with_review:
        print("\n==> Human review UI (--with-review). Close when done.")
        _run("scripts.review")
    else:
        auto_args: list[str] = []
        if args.strict_vision:
            auto_args.append("--strict-vision")
        if args.require_pdf_text:
            auto_args.append("--require-pdf-text")
        _run("scripts.auto_accept", auto_args)

    if not args.skip_export:
        export_args = ["--validation-package"]
        if args.allow_shortfall and not args.strict_quota:
            export_args.append("--allow-shortfall")
        try:
            _run("scripts.export", export_args)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 2:
                print(
                    "Export blocked (quota / gates). "
                    "Auto mode usually uses --allow-shortfall; "
                    "or mine more sources / drop --strict-quota."
                )
            raise

    print("\nAutomatic pipeline complete.")
    print("Package: data/export/marathi_ocr_validation_100/")


if __name__ == "__main__":
    main()
