from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery.download import download_many
from discovery.search import DEFAULT_SEARCH_SEEDS, discover_pdf_candidates
from llm.client import LLMClient
from pipeline.config import load_config, resolve_path
from pipeline.env_loader import load_dotenv
from pipeline.ingest_render import ingest_file
from pipeline.io_utils import read_json, write_json
from pipeline.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def _ingested_urls(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    try:
        manifest = read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return set()
    urls: set[str] = set()
    for src in manifest.get("sources") or []:
        u = (src.get("source_url") or "").strip()
        if u:
            urls.add(u)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-discover Marathi PDFs (portals + built-in web search seeds; no user query needed)"
    )
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--no-portals", action="store_true", help="Skip seed portal crawl")
    parser.add_argument(
        "--no-ddg",
        action="store_true",
        help="Skip DuckDuckGo (enabled by default)",
    )
    parser.add_argument("--no-llm-rank", action="store_true", help="Skip optional Kimi ranking")
    parser.add_argument("--no-ingest", action="store_true", help="Download only; skip raw/ ingest")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Do not skip URLs already in the source manifest (default: skip)",
    )
    parser.add_argument("--doc-type", default="government_resolution")
    parser.add_argument(
        "--exclude-list",
        default="",
        help="Path to exclude list (default: sources/exclude_sources.txt)",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config()
    setup_logging(cfg)

    sources = resolve_path(cfg, "sources")
    downloads = sources / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    manifest = sources / "manifest.json"
    exclude = args.exclude_list or str(sources / "exclude_sources.txt")
    use_ddg = bool(cfg.get("use_ddg_by_default", True)) and not args.no_ddg
    llm_rank = bool(cfg.get("llm_discover_rank", True)) and not args.no_llm_rank
    seeds = cfg.get("discover_search_seeds") or DEFAULT_SEARCH_SEEDS
    if isinstance(seeds, str):
        seeds = [seeds]

    already = set() if args.include_existing else _ingested_urls(manifest)
    # Ask discovery for a larger ranked pool so skip-existing still yields max-docs
    pool = max(args.max_docs * 5, args.max_docs + len(already) + 8)

    client = LLMClient.from_config(cfg)
    discovery = discover_pdf_candidates(
        None,
        client,
        max_urls=pool,
        use_portals=not args.no_portals,
        use_ddg=use_ddg,
        llm_rank=llm_rank,
        exclude_path=exclude,
        search_seeds=list(seeds),
    )
    ranked = discovery.get("ranked") or []
    urls = [r["url"] for r in ranked if r.get("url")]
    write_json(sources / "discovery_last.json", discovery)
    logger.info(
        "Discovery auto: portals=%s ddg=%s seeds=%s kimi=%s excluded=%s candidates=%s ranked=%s skip_existing=%s",
        discovery.get("portal_count"),
        discovery.get("web_search_count"),
        len(discovery.get("search_seeds") or []),
        discovery.get("kimi_url_count"),
        discovery.get("excluded_count"),
        discovery.get("candidate_count"),
        len(urls),
        len(already),
    )

    results = download_many(
        urls,
        downloads,
        max_docs=args.max_docs,
        skip_urls=already,
    )
    ok_paths = [r for r in results if r.get("ok") and r.get("path") and not r.get("deduped") and not r.get("skipped")]
    # Also ingest deduped hits that map to on-disk PDFs not yet in manifest
    for r in results:
        if r.get("ok") and r.get("path") and r.get("deduped") and r.get("url") not in already:
            if r not in ok_paths:
                ok_paths.append(r)
    write_json(
        sources / "discovery_downloads.json",
        {
            "mode": "auto",
            "search_seeds": discovery.get("search_seeds"),
            "downloads": results,
            "ok": len(ok_paths),
            "skipped_existing": len(already),
        },
    )

    ingested = 0
    if not args.no_ingest:
        raw = resolve_path(cfg, "raw")
        for item in ok_paths:
            try:
                ingest_file(
                    item["path"],
                    raw,
                    manifest,
                    source_url=item.get("url", ""),
                    document_type=args.doc_type,
                )
                ingested += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ingest failed %s: %s", item.get("path"), exc)

    print(
        f"Discover done: downloaded_ok={len(ok_paths)} ingested={ingested} "
        f"skipped_existing={len(already)} dir={downloads} report={sources / 'discovery_last.json'}"
    )
    if discovery.get("llm_error"):
        print(
            f"Note: Kimi rank soft-failed ({discovery['llm_error'][:160]}); "
            "portal/DDG results still used."
        )
    if ingested == 0 and args.max_docs > 0:
        print(
            "WARNING: no new documents ingested this round "
            "(sources exhausted, portals blocked, or all candidates already present)."
        )


if __name__ == "__main__":
    main()
