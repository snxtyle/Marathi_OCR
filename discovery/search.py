"""Crawl-first discovery of public Marathi government PDFs.

Primary: Maharashtra portal crawl + optional DuckDuckGo.
Optional: Kimi rank/classify only (not the sole crawler).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, unquote, urljoin, urlparse

import requests
from llm.client import LLMClient

logger = logging.getLogger(__name__)

USER_AGENT = "MarathiOCR-DataFactory/1.0 (+research; respectful crawl)"

SEED_PORTALS = [
    "https://gr.maharashtra.gov.in/",
    "https://gr.maharashtra.gov.in/Site/Home/Index.aspx",
    "https://gr.maharashtra.gov.in/Site/Upload/Government%20Resolutions/Marathi/",
    "https://www.maharashtra.gov.in/",
    "https://lj.maharashtra.gov.in/",
    "https://lj.maharashtra.gov.in/Sitemap/lj/pdf/",
]

# Built-in web-search seeds — user never needs to pass --query
DEFAULT_SEARCH_SEEDS = [
    "महाराष्ट्र शासन निर्णय मराठी filetype:pdf",
    "महाराष्ट्र अधिसूचना मराठी filetype:pdf",
    "महाराष्ट्र परिपत्रक मराठी filetype:pdf",
    "maharashtra government resolution marathi filetype:pdf",
    "site:gr.maharashtra.gov.in मराठी filetype:pdf",
]

PDF_HREF_RE = re.compile(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', re.IGNORECASE)
UDDG_RE = re.compile(r"[?&]uddg=([^&\"']+)", re.IGNORECASE)
HTTP_PDF_RE = re.compile(r"https?://[^\s\"'<>]+\.pdf(?:\?[^\s\"'<>]*)?", re.IGNORECASE)


def _is_probably_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or ".pdf?" in url.lower()


def load_exclude_list(path: str | Path | None) -> set[str]:
    """Load URLs/filenames that must not be reused for new validation mining."""
    out: set[str] = set()
    if not path:
        return out
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s.lower())
    return out


def is_excluded(url: str, excludes: set[str]) -> bool:
    if not excludes:
        return False
    u = (url or "").lower()
    name = Path(urlparse(url).path).name.lower()
    return u in excludes or name in excludes or any(x in u for x in excludes if len(x) > 8)


def fetch_html(url: str, *, timeout: int = 30) -> str:
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        timeout=timeout,
        allow_redirects=True,
    )
    resp.raise_for_status()
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ctype:
        return ""
    return resp.text


def extract_pdf_links(page_url: str, html: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in PDF_HREF_RE.finditer(html or ""):
        href = match.group(1).strip()
        abs_url = urljoin(page_url, href)
        if abs_url in seen or not abs_url.startswith("http"):
            continue
        if not _is_probably_pdf_url(abs_url):
            continue
        seen.add(abs_url)
        found.append({"url": abs_url, "title": "", "source_page": page_url})
    for match in HTTP_PDF_RE.finditer(html or ""):
        abs_url = match.group(0).rstrip(").,;]")
        if abs_url in seen:
            continue
        seen.add(abs_url)
        found.append({"url": abs_url, "title": "", "source_page": page_url})
    return found


def web_search_pdfs(query: str, *, max_results: int = 20) -> list[dict[str, str]]:
    """DuckDuckGo HTML search restricted to PDF filetype."""
    q = query if "filetype:pdf" in query.lower() else f"{query} filetype:pdf"
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"
    try:
        html = fetch_html(url, timeout=40)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []

    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in UDDG_RE.finditer(html or ""):
        target = unquote(match.group(1))
        if not target.startswith("http") or target in seen:
            continue
        if not _is_probably_pdf_url(target):
            continue
        seen.add(target)
        found.append({"url": target, "title": "", "source_page": "duckduckgo"})
        if len(found) >= max_results:
            break
    for item in extract_pdf_links(url, html):
        if item["url"] not in seen:
            seen.add(item["url"])
            found.append({**item, "source_page": "duckduckgo"})
        if len(found) >= max_results:
            break
    logger.info("DuckDuckGo PDF hits: %d", len(found))
    return found


def crawl_seed_portals(
    portals: list[str] | None = None,
    *,
    delay_s: float = 0.8,
) -> list[dict[str, str]]:
    portals = portals or SEED_PORTALS
    all_links: list[dict[str, str]] = []
    for portal in portals:
        try:
            html = fetch_html(portal)
            links = extract_pdf_links(portal, html)
            logger.info("Portal %s -> %d pdf links", portal, len(links))
            all_links.extend(links)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Portal crawl failed %s: %s", portal, exc)
        time.sleep(delay_s)
    dedup: dict[str, dict[str, str]] = {}
    for item in all_links:
        dedup[item["url"]] = item
    return list(dedup.values())


def discover_pdf_candidates(
    query: str | None = None,
    client: LLMClient | None = None,
    *,
    max_urls: int = 20,
    use_portals: bool = True,
    use_ddg: bool = True,
    llm_rank: bool = True,
    exclude_path: str | Path | None = None,
    search_seeds: list[str] | None = None,
) -> dict[str, Any]:
    """Crawl-first discovery with built-in seeds — no user query required.

    Optional ``query`` is only an extra seed if provided; default path uses
    portals + ``DEFAULT_SEARCH_SEEDS`` (or config seeds).
    """
    excludes = load_exclude_list(exclude_path)
    candidates: list[dict[str, str]] = []
    seeds = list(search_seeds or DEFAULT_SEARCH_SEEDS)
    if query and query.strip() and query.strip() not in seeds:
        seeds.insert(0, query.strip())

    portal_links: list[dict[str, str]] = []
    if use_portals:
        portal_links = crawl_seed_portals()
        candidates.extend(portal_links)

    search_links: list[dict[str, str]] = []
    if use_ddg:
        per_seed = max(5, (max_urls + len(seeds) - 1) // max(len(seeds), 1))
        seen_search: set[str] = set()
        for seed in seeds:
            for item in web_search_pdfs(seed, max_results=per_seed):
                u = item.get("url", "")
                if u and u not in seen_search:
                    seen_search.add(u)
                    search_links.append(item)
            time.sleep(0.4)
        candidates.extend(search_links)

    # Optional: Kimi propose/rank using first seed (assist only)
    kimi_links: list[dict[str, str]] = []
    primary = seeds[0] if seeds else "मराठी शासन PDF"
    llm_result: dict[str, Any] = {"ok": False, "pdf_urls": [], "queries": seeds}
    if llm_rank and client is not None and getattr(client, "enabled", False):
        llm_result = client.propose_pdf_urls(primary, max_urls=max(5, max_urls // 2))
        for url in llm_result.get("pdf_urls") or []:
            if isinstance(url, str) and url.startswith("http"):
                kimi_links.append({"url": url, "title": "", "source_page": "kimi"})
        candidates.extend(kimi_links)

    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    excluded_n = 0
    for c in candidates:
        u = c.get("url", "")
        if not u or u in seen:
            continue
        if is_excluded(u, excludes):
            excluded_n += 1
            continue
        seen.add(u)
        unique.append(c)

    if llm_rank and client is not None and getattr(client, "enabled", False) and unique:
        ranked = client.rank_pdf_candidates(unique, max_keep=max_urls)
    else:
        ranked = [
            {"url": c.get("url", ""), "title": c.get("title", ""), "score": 0.5}
            for c in unique[:max_urls]
        ]

    return {
        "query": primary,
        "search_seeds": seeds,
        "llm_ok": bool(llm_result.get("ok")),
        "llm_error": llm_result.get("error"),
        "llm_notes": llm_result.get("notes", ""),
        "llm_queries": llm_result.get("queries") or seeds,
        "kimi_url_count": len(kimi_links),
        "web_search_count": len(search_links),
        "portal_count": len(portal_links),
        "excluded_count": excluded_n,
        "candidate_count": len(unique),
        "ranked": ranked,
        "discovery_mode": "crawl_first_auto",
    }
