"""Download discovered PDFs with checksum + provenance."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "MarathiOCR-DataFactory/1.0 (+research; respectful download)"


def _safe_filename(url: str, index: int) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name or f"doc_{index:04d}.pdf"
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    if len(name) > 120:
        stem = name[:-4][:100]
        name = f"{stem}.pdf"
    return f"{index:04d}_{name}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_pdf(
    url: str,
    dest_dir: str | Path,
    *,
    index: int = 1,
    timeout: int = 120,
    min_bytes: int = 5_000,
) -> dict[str, Any]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"},
        timeout=timeout,
        allow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.content
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "html" in ctype and not data[:8].startswith(b"%PDF"):
        raise ValueError(f"Not a PDF (content-type={ctype}): {url}")
    if not data.startswith(b"%PDF") and b"%PDF" not in data[:1024]:
        # soft check
        if len(data) < min_bytes:
            raise ValueError(f"Download too small / not PDF: {url}")
    digest = sha256_bytes(data)
    # Dedupe by hash filename suffix
    fname = _safe_filename(url, index)
    out = dest_dir / fname
    # If same hash already present under another name, reuse
    for existing in dest_dir.glob("*.pdf"):
        if sha256_bytes(existing.read_bytes()) == digest:
            return {
                "ok": True,
                "url": url,
                "path": str(existing),
                "sha256": digest,
                "bytes": existing.stat().st_size,
                "deduped": True,
            }
    out.write_bytes(data)
    return {
        "ok": True,
        "url": url,
        "path": str(out),
        "sha256": digest,
        "bytes": len(data),
        "deduped": False,
    }


def download_many(
    urls: list[str],
    dest_dir: str | Path,
    *,
    max_docs: int = 20,
    delay_s: float = 1.0,
    skip_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Download until ``max_docs`` successful PDFs or URL list exhausted.

    Failed / skipped / already-seen hashes do not count toward the success quota,
    so callers actually get up to ``max_docs`` new files when the pool allows.
    """
    results: list[dict[str, Any]] = []
    seen_hash: set[str] = set()
    skip = {u.strip().lower() for u in (skip_urls or set()) if u}
    ok_new = 0
    i = 0
    for url in urls:
        if ok_new >= max_docs:
            break
        i += 1
        if url.strip().lower() in skip:
            results.append({"ok": False, "url": url, "skipped": True, "error": "already_ingested"})
            continue
        try:
            meta = download_pdf(url, dest_dir, index=i)
            if meta["sha256"] in seen_hash or meta.get("deduped"):
                meta["ok"] = True
                meta["deduped"] = True
                results.append(meta)
                # Deduped against prior download — not a new doc for this batch
                continue
            seen_hash.add(meta["sha256"])
            results.append(meta)
            ok_new += 1
            logger.info("Downloaded %s -> %s (%s bytes)", url, meta["path"], meta["bytes"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download failed %s: %s", url, exc)
            results.append({"ok": False, "url": url, "error": str(exc)})
        time.sleep(delay_s)
    return results
