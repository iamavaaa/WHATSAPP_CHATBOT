"""
Crawl COMMANDO Networks public site, extract main text, dedupe, write JSONL.

Parsing stack (crawling + text extraction):
  - Link discovery: BeautifulSoup with the built-in ``html.parser`` (no lxml).
  - Main text: trafilatura.extract first; if empty, BeautifulSoup strips noise tags
    and reads main/article/body text as a fallback.

Output format matches the rest of the project: one JSON object per line with a "text" field.

Before running:
- Review https://commandonetworks.com/ terms of use and robots.txt.
- URLs are seeded from the live sitemap (see commando_sitemap.py) plus link BFS.
- Prefer a small page limit first (e.g. MAX_PAGES=50) to validate quality.

For ~2.5GB raw + ~50MB cleaned corpus (assignment-style), use instead:
  python scripts/prepare_commando_dataset.py

Usage (from repo root):
  pip install -r requirements.txt
  cd scripts
  python crawl_commando_site.py
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import mmh3
import requests
import trafilatura
from bs4 import BeautifulSoup

from commando_sitemap import discover_page_urls_from_sitemaps

# Built-in parser only (no lxml dependency).
HTML_PARSER = "html.parser"
_BS_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside"]

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = BASE_DIR / "data" / "commando_networks.jsonl"

# Same site only; adjust if you add a sitemap-only mode later.
SEED_URL = "https://commandonetworks.com/"
MAX_PAGES = int(os.environ.get("CRAWL_MAX_PAGES", "120"))
SITEMAP_CAP = int(os.environ.get("COMMANDO_SITEMAP_CAP", "5000"))
REQUEST_DELAY_SEC = float(os.environ.get("CRAWL_DELAY_SEC", "1.0"))
MIN_TEXT_LEN = 80
USER_AGENT = "CommandoRAGBot/1.0 (+educational; contact: your-email)"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _host_key(netloc: str) -> str:
    """Treat www.example.com and example.com as the same site."""
    host = netloc.lower().split("@")[-1]  # strip userinfo if ever present
    if host.startswith("www."):
        host = host[4:]
    return host


def _same_site(url: str, base_netloc: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    return _host_key(p.netloc) == _host_key(base_netloc)


def _normalize_url(url: str) -> str:
    url, _frag = urldefrag(url)
    if url.endswith("/") and len(urlparse(url).path) > 1:
        url = url.rstrip("/")
    return url


def _clean_text(raw: str) -> str:
    t = re.sub(r"\s+", " ", raw).strip()
    return t


def _robots_allowed(rp: RobotFileParser | None, url: str) -> bool:
    if rp is None:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def fetch_robots(base: str) -> RobotFileParser | None:
    parsed = urlparse(base)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp
    except Exception:
        return None


def _fallback_text_bs4(html: str) -> str:
    """When trafilatura returns nothing, strip chrome tags and take main/article/body."""
    soup = BeautifulSoup(html, HTML_PARSER)
    for tag in soup.find_all(_BS_NOISE_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return ""
    raw = main.get_text(separator="\n", strip=True)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return _clean_text(raw)


def extract_text(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted:
        return _clean_text(extracted)
    return _fallback_text_bs4(html)


def crawl() -> None:
    base_netloc = urlparse(SEED_URL).netloc
    robots = fetch_robots(SEED_URL)

    seen_urls: set[str] = set()
    seen_hashes: set[int] = set()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Prefer live sitemap URLs so product/detail pages are not missed by link-only BFS.
    sitemap_urls = discover_page_urls_from_sitemaps(
        _session,
        SEED_URL,
        delay=REQUEST_DELAY_SEC,
        max_pages=SITEMAP_CAP,
    )
    queue_init: list[str] = []
    qi: set[str] = set()
    for u in sitemap_urls:
        nu = _normalize_url(u)
        if nu not in qi:
            qi.add(nu)
            queue_init.append(nu)
    seed_n = _normalize_url(SEED_URL)
    if seed_n not in qi:
        queue_init.insert(0, seed_n)
    queue: deque[str] = deque(queue_init)

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        while queue and count < MAX_PAGES:
            url = queue.popleft()
            url = _normalize_url(url)
            if url in seen_urls:
                continue
            if not _same_site(url, base_netloc):
                continue
            if not _robots_allowed(robots, url):
                continue

            seen_urls.add(url)
            time.sleep(REQUEST_DELAY_SEC)

            try:
                r = _session.get(url, timeout=30)
                r.raise_for_status()
            except Exception:
                continue

            ctype = (r.headers.get("content-type") or "").lower()
            if "text/html" not in ctype and "application/xhtml" not in ctype:
                continue

            text = extract_text(r.text, url)
            if len(text) < MIN_TEXT_LEN:
                continue

            h = mmh3.hash128(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            line = json.dumps({"text": text, "source_url": url}, ensure_ascii=False)
            out.write(line + "\n")
            count += 1

            soup = BeautifulSoup(r.text, HTML_PARSER)
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                next_url = urljoin(url, href)
                next_url = _normalize_url(next_url)
                if _same_site(next_url, base_netloc) and next_url not in seen_urls:
                    queue.append(next_url)

    print(f"Wrote {count} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    crawl()
