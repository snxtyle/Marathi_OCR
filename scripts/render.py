from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import load_config, resolve_path
from pipeline.ingest_render import render_source
from pipeline.io_utils import read_json, write_json
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ingested sources to high-res PNG pages")
    parser.add_argument("--dpi", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=0, help="Cap pages per source (0=all)")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    dpi = args.dpi or int(cfg.get("render_dpi", 300))
    sources = resolve_path(cfg, "sources")
    rendered = resolve_path(cfg, "rendered")
    manifest = read_json(sources / "manifest.json")
    pages = []
    for src in manifest.get("sources", []):
        pages.extend(
            render_source(
                src,
                rendered,
                dpi=dpi,
                max_pages=args.max_pages or None,
            )
        )
    write_json(rendered / "pages.json", {"pages": pages, "dpi": dpi})
    print(f"Rendered {len(pages)} pages at {dpi} DPI -> {rendered}")


if __name__ == "__main__":
    main()
