from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.io_utils import read_jsonl, write_jsonl
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def seed_benchmark(src_dataset: Path, bench_dir: Path) -> int:
    images_src = src_dataset / "images"
    labels_src = src_dataset / "validation.jsonl"
    images_dst = bench_dir / "images"
    images_dst.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(labels_src)
    out = []
    for rec in records:
        name = Path(rec["image_filename"]).name
        src = images_src / name
        if not src.is_file():
            logger.warning("Missing image %s", src)
            continue
        shutil.copy2(src, images_dst / name)
        row = {
            "image_filename": f"images/{name}",
            "expected_text": rec["expected_text"],
            "issue_type": rec.get("issue_type", ""),
        }
        if rec.get("source_id"):
            row["source_id"] = rec["source_id"]
        # Derive digit flag for slicing when not present in the seed JSONL
        expected = rec.get("expected_text") or ""
        if "contains_marathi_digits" in rec:
            row["contains_marathi_digits"] = bool(rec["contains_marathi_digits"])
        else:
            row["contains_marathi_digits"] = any("०" <= ch <= "९" for ch in expected)
        out.append(row)
    write_jsonl(bench_dir / "labels.jsonl", out)
    logger.info("Seeded %d benchmark samples into %s", len(out), bench_dir)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed OCR benchmark from curated dataset")
    parser.add_argument(
        "--src",
        default=str(Path.home() / "Desktop" / "marathi_ocr_validation_100"),
        help="Source curated dataset directory",
    )
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg)
    bench = resolve_path(cfg, "benchmark")
    n = seed_benchmark(Path(args.src), bench)
    print(f"Seeded {n} benchmark labels")


if __name__ == "__main__":
    main()
