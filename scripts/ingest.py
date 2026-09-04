from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.ingest_render import ingest_file
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest source documents into raw/")
    parser.add_argument("paths", nargs="+", help="PDF/PNG/JPG/TIFF files or directories")
    parser.add_argument("--url", default="", help="Optional source URL for provenance")
    parser.add_argument("--doc-type", default="government_resolution")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    raw = resolve_path(cfg, "raw")
    sources = resolve_path(cfg, "sources")
    manifest = sources / "manifest.json"

    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                files.extend(path.glob(ext))
        else:
            files.append(path)

    for f in files:
        ingest_file(f, raw, manifest, source_url=args.url, document_type=args.doc_type)
    print(f"Ingested {len(files)} file(s). Manifest: {manifest}")


if __name__ == "__main__":
    main()
