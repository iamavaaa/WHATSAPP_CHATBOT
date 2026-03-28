"""
Discover page URLs from the live COMMANDO site via robots.txt Sitemap lines and sitemap.xml.

Uses regex for <loc> so xmlns variants still parse. Respects same-host filtering.
"""

from __future__ import annotations

import re
import time
from collections import deque
from urllib.parse import urldefrag, urlparse

import requests

_LOC_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.IGNORECASE)


def _host_key(netloc: str) -> str:
    host = netloc.lower().split("@")[-1]
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_url(url: str) -> str:
    url, _frag = urldefrag(url.strip())
    if url.endswith("/") and len(urlparse(url).path) > 1:
        url = url.rstrip("/")
    return url


def _same_site(url: str, base_host_key: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    return _host_key(p.netloc) == base_host_key


def _sitemap_entry_points(session: requests.Session, base_url: str, delay: float) -> list[str]:
    parsed = urlparse(base_url)
    scheme, netloc = parsed.scheme, parsed.netloc
    candidates = [f"{scheme}://{netloc}/sitemap.xml"]
    robots_url = f"{scheme}://{netloc}/robots.txt"
    time.sleep(delay)
    try:
        r = session.get(robots_url, timeout=30)
        r.raise_for_status()
        for line in r.text.splitlines():
            line = line.strip()
            if line.lower().startswith("sitemap:"):
                u = line.split(":", 1)[1].strip()
                if u:
                    candidates.append(u)
    except Exception:
        pass
    # Dedupe, preserve order
    out: list[str] = []
    seen: set[str] = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _is_sitemap_index(content: str) -> bool:
    return "sitemapindex" in content[:4000].lower()


def discover_page_urls_from_sitemaps(
    session: requests.Session,
    base_url: str,
    *,
    delay: float,
    max_pages: int,
) -> list[str]:
    """
    Return normalized https URLs on the same host as base_url, up to max_pages.
    """
    parsed = urlparse(base_url)
    base_host_key = _host_key(parsed.netloc)

    sitemap_queue: deque[str] = deque(_sitemap_entry_points(session, base_url, delay))
    seen_sitemaps: set[str] = set()
    page_urls: list[str] = []
    seen_pages: set[str] = set()

    while sitemap_queue and len(page_urls) < max_pages:
        sm_url = sitemap_queue.popleft().strip()
        if sm_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sm_url)

        time.sleep(delay)
        try:
            r = session.get(sm_url, timeout=60)
            r.raise_for_status()
        except Exception:
            continue

        ctype = (r.headers.get("content-type") or "").lower()
        head = (r.text or "")[:4000].lower()
        if "xml" not in ctype and not sm_url.lower().endswith(".xml"):
            if "<?xml" not in head and "<urlset" not in head and "sitemapindex" not in head:
                continue

        content = r.text
        locs = [u.strip() for u in _LOC_RE.findall(content) if u.strip()]

        if _is_sitemap_index(content):
            for u in locs:
                if u not in seen_sitemaps:
                    sitemap_queue.append(u)
        else:
            for u in locs:
                if not _same_site(u, base_host_key):
                    continue
                nu = _normalize_url(u)
                if nu in seen_pages:
                    continue
                seen_pages.add(nu)
                page_urls.append(nu)
                if len(page_urls) >= max_pages:
                    break

    return page_urls
