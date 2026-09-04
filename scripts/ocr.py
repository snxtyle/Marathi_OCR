from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from ocr.factory import create_backend
from pipeline.cache import OCRCache, ResumeStore
from pipeline.config import load_config, resolve_path
from pipeline.hashing import sha256_file
from pipeline.io_utils import read_jsonl, write_jsonl
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR pre-label candidates (cached)")
    parser.add_argument("--backend", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    backend_name = args.backend or cfg.get("ocr_backend", "tesseract")
    cand_dir = resolve_path(cfg, "candidates")
    cache_dir = resolve_path(cfg, "cache") / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = OCRCache(cache_dir)
    resume = ResumeStore(cache_dir / f"resume_{backend_name}.txt") if cfg.get("resume", True) else None

    records = read_jsonl(cand_dir / "candidates.jsonl")
    if args.limit:
        records = records[: args.limit]

    backend = create_backend(backend_name, use_gpu=False, lang=str(cfg.get("ocr_lang") or "mar"))

    for rec in records:
        cid = rec["id"]
        if resume and resume.is_done(cid) and rec.get("ocr_prediction"):
            continue
        img_path = ROOT / rec["image_filename"]
        if not img_path.is_file():
            img_path = Path(rec.get("image_path", ""))
        if not img_path.is_file():
            logger.warning("Missing image for %s", cid)
            continue
        digest = sha256_file(img_path)
        cached = cache.get(digest, backend.name, backend.model_version)
        if cached and "text" in cached:
            text = cached["text"]
        else:
            with Image.open(img_path) as img:
                text = backend.recognize(img)
            cache.set(digest, backend.name, backend.model_version, {"text": text, "id": cid})
        rec["ocr_prediction"] = text
        rec["ocr_backend"] = backend.name
        rec["ocr_model_version"] = backend.model_version
        rec["sha256"] = digest
        # Never treat OCR as ground truth. Prefer PDF text assist when present.
        if not rec.get("expected_text"):
            rec["expected_text"] = rec.get("text_assist") or ""
        if resume:
            resume.mark_done(cid)

    write_jsonl(cand_dir / "candidates.jsonl", records)
    write_jsonl(cand_dir / "candidates_ocr.jsonl", records)
    print(f"OCR pre-labeled {len(records)} candidates with {backend.name}")


if __name__ == "__main__":
    main()
