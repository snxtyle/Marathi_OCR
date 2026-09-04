from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fitz
from PIL import Image

from pipeline.io_utils import read_json, write_json

logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _source_id_from_path(path: Path) -> str:
    stem = path.stem
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{stem}_{digest}"


def ingest_file(
    src: str | Path,
    raw_dir: str | Path,
    manifest_path: str | Path,
    *,
    source_url: str = "",
    document_type: str = "unknown",
) -> dict[str, Any]:
    src = Path(src)
    if src.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type: {src.suffix}")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)

    source_id = _source_id_from_path(dest)
    page_count = 1
    if dest.suffix.lower() == ".pdf":
        doc = fitz.open(dest)
        page_count = len(doc)
        doc.close()

    entry = {
        "source_id": source_id,
        "source_path": str(dest),
        "source_url": source_url or "",
        "document_type": document_type,
        "filename": dest.name,
        "page_count": page_count,
        "ext": dest.suffix.lower(),
    }

    manifest_path = Path(manifest_path)
    manifest: dict[str, Any] = {"sources": []}
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
    sources = [s for s in manifest.get("sources", []) if s.get("source_id") != source_id]
    sources.append(entry)
    manifest["sources"] = sources
    write_json(manifest_path, manifest)
    logger.info("Ingested %s as %s (%d pages)", dest.name, source_id, page_count)
    return entry


def render_source(
    source: dict[str, Any],
    rendered_dir: str | Path,
    *,
    dpi: int = 300,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    rendered_dir = Path(rendered_dir)
    rendered_dir.mkdir(parents=True, exist_ok=True)
    path = Path(source["source_path"])
    source_id = source["source_id"]
    pages_meta: list[dict[str, Any]] = []

    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        n = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(n):
            pix = doc[i].get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            out = rendered_dir / f"{source_id}_p{i + 1:03d}.png"
            pix.save(out)
            pages_meta.append(
                {
                    "source_id": source_id,
                    "source_path": str(path),
                    "source_url": source.get("source_url", ""),
                    "page_number": i + 1,
                    "document_type": source.get("document_type", "unknown"),
                    "rendered_path": str(out),
                    "dpi": dpi,
                    "width": pix.width,
                    "height": pix.height,
                }
            )
        doc.close()
    else:
        with Image.open(path) as img:
            img = img.convert("RGB")
            out = rendered_dir / f"{source_id}_p001.png"
            img.save(out, format="PNG", dpi=(dpi, dpi))
            pages_meta.append(
                {
                    "source_id": source_id,
                    "source_path": str(path),
                    "source_url": source.get("source_url", ""),
                    "page_number": 1,
                    "document_type": source.get("document_type", "unknown"),
                    "rendered_path": str(out),
                    "dpi": dpi,
                    "width": img.width,
                    "height": img.height,
                }
            )
    return pages_meta
