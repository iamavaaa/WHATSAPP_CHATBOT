"""
Build a large raw crawl from https://commandonetworks.com/ then clean down to ~50MB JSONL.

Parsing stack:
  - URL discovery: BeautifulSoup + ``html.parser`` (same as crawl_commando_site.py).
  - Cleaning raw HTML: trafilatura.extract first; if empty, BeautifulSoup fallback on
    main/article/body after stripping noise tags.

Mirrors the Common Crawl workflow in this repo:
  1) Collect ~2–3GB of raw material (full HTML per fetch, multiple passes over the site).
  2) Clean, quality-filter, dedupe, write ~50MB of {"text": ...} lines for embeddings/RAG.

Respect robots.txt and rate limits. Review site terms before running.

Usage (from repo root):
  pip install -r requirements.txt
  python scripts/prepare_commando_dataset.py
  python scripts/prepare_commando_dataset.py --clean-only   # reuse existing raw file
  python scripts/prepare_commando_dataset.py --raw-only     # only grow raw corpus

Env:
  COMMANDO_TARGET_RAW_GB   default 2.5
  COMMANDO_SAMPLE_MB       default 50
  CRAWL_DELAY_SEC          delay between requests (default 0.5)
  COMMANDO_MAX_PASSES      safety cap on full-site repeat passes (default 800)
  COMMANDO_DISCOVERY_CAP   max unique URLs to discover (default 2500)
  COMMANDO_SITEMAP_CAP   max URLs pulled from sitemap.xml / robots Sitemap (default 5000)
"""

from __future__ import annotations

import argparse
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
from tqdm import tqdm

from commando_sitemap import discover_page_urls_from_sitemaps

HTML_PARSER = "html.parser"
_BS_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside"]

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw_commando_crawl.jsonl"
SAMPLE_PATH = BASE_DIR / "data" / "commando_networks.jsonl"

SEED_URL = "https://commandonetworks.com/"
USER_AGENT = "CommandoRAGBot/1.0 (+educational; contact: your-email)"

TARGET_RAW_BYTES = float(os.environ.get("COMMANDO_TARGET_RAW_GB", "2.5")) * 1024**3
TARGET_SAMPLE_BYTES = float(os.environ.get("COMMANDO_SAMPLE_MB", "50")) * 1024**2
REQUEST_DELAY_SEC = float(os.environ.get("CRAWL_DELAY_SEC", "0.5"))
MAX_PASSES = int(os.environ.get("COMMANDO_MAX_PASSES", "800"))
DISCOVERY_CAP = int(os.environ.get("COMMANDO_DISCOVERY_CAP", "2500"))
SITEMAP_CAP = int(os.environ.get("COMMANDO_SITEMAP_CAP", "5000"))

MIN_TEXT_LEN_CLEAN = 200
MIN_TEXT_LEN_RAW = 50  # still store HTML even if extract is tiny; HTML carries bytes

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _host_key(netloc: str) -> str:
    host = netloc.lower().split("@")[-1]
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
    return re.sub(r"\s+", " ", raw).strip()


def _robots_allowed(rp: RobotFileParser | None, url: str) -> bool:
    if rp is None:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def _fallback_text_bs4(html: str) -> str:
    soup = BeautifulSoup(html, HTML_PARSER)
    for tag in soup.find_all(_BS_NOISE_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return ""
    raw = main.get_text(separator="\n", strip=True)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return _clean_text(raw)


def extract_text_from_html(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html,
        url=url or SEED_URL,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if extracted:
        return _clean_text(extracted)
    return _fallback_text_bs4(html)


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


def discover_urls() -> list[str]:
    """List same-host HTML URLs: sitemap first, then BFS link crawl (capped)."""
    base_netloc = urlparse(SEED_URL).netloc
    robots = fetch_robots(SEED_URL)
    seen: set[str] = set()
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
    ordered: list[str] = []

    with tqdm(desc="Discovering URLs", unit="url") as pbar:
        while queue and len(ordered) < DISCOVERY_CAP:
            url = _normalize_url(queue.popleft())
            if url in seen:
                continue
            if not _same_site(url, base_netloc) or not _robots_allowed(robots, url):
                continue

            time.sleep(REQUEST_DELAY_SEC)

            try:
                r = _session.get(url, timeout=45)
                r.raise_for_status()
            except Exception:
                continue

            ctype = (r.headers.get("content-type") or "").lower()
            if "text/html" not in ctype and "application/xhtml" not in ctype:
                continue

            seen.add(url)
            ordered.append(url)
            pbar.update(1)

            soup = BeautifulSoup(r.text, HTML_PARSER)
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                nxt = _normalize_url(urljoin(url, href))
                if _same_site(nxt, base_netloc) and nxt not in seen:
                    queue.append(nxt)

    return ordered


def build_raw_corpus(urls: list[str]) -> None:
    """Append full HTML lines until RAW_PATH reaches TARGET_RAW_BYTES (multi-pass)."""
    if not urls:
        raise RuntimeError("No URLs discovered; cannot build raw corpus.")

    base_netloc = urlparse(SEED_URL).netloc
    robots = fetch_robots(SEED_URL)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists():
        RAW_PATH.unlink()

    pass_id = 0
    size_written = 0
    pbar = tqdm(total=int(TARGET_RAW_BYTES), unit="B", unit_scale=True, desc="Raw crawl (to target size)")

    with open(RAW_PATH, "a", encoding="utf-8") as raw_out:
        while size_written < TARGET_RAW_BYTES and pass_id < MAX_PASSES:
            pass_id += 1
            for url in urls:
                if size_written >= TARGET_RAW_BYTES:
                    break
                if not _robots_allowed(robots, url):
                    continue
                time.sleep(REQUEST_DELAY_SEC)
                try:
                    r = _session.get(url, timeout=45)
                    r.raise_for_status()
                except Exception:
                    continue

                ctype = (r.headers.get("content-type") or "").lower()
                if "text/html" not in ctype and "application/xhtml" not in ctype:
                    continue

                html = r.text
                if len(html) < MIN_TEXT_LEN_RAW:
                    continue

                record = {
                    "url": url,
                    "pass": pass_id,
                    "content_type": r.headers.get("content-type", ""),
                    "html": html,
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                raw_out.write(line)
                raw_out.flush()
                b = len(line.encode("utf-8"))
                size_written += b
                pbar.update(b)

        pbar.close()

    final = RAW_PATH.stat().st_size
    print(f"Raw corpus: {final / (1024**3):.2f} GB at {RAW_PATH}")
    if final < TARGET_RAW_BYTES * 0.9:
        print(
            "Warning: site may not yield enough unique HTML to reach the target in fewer passes. "
            "Increase COMMANDO_MAX_PASSES or COMMANDO_TARGET_RAW_GB, or confirm more URLs are reachable."
        )


def is_quality_content(text: str) -> bool:
    if len(text) < MIN_TEXT_LEN_CLEAN:
        return False
    if text.count("{") > 5 or "<html" in text.lower():
        return False
    return True


def clean_to_sample() -> None:
    """Stream raw JSONL -> deduped text JSONL up to TARGET_SAMPLE_BYTES."""
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}. Run without --clean-only first.")

    seen_hashes: set[int] = set()
    sample_bytes = 0
    rows_out = 0

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SAMPLE_PATH.exists():
        SAMPLE_PATH.unlink()

    with open(RAW_PATH, "r", encoding="utf-8") as raw_in, open(
        SAMPLE_PATH, "w", encoding="utf-8"
    ) as out, tqdm(desc="Cleaning to sample", unit="B", unit_scale=True) as pbar:
        for line in raw_in:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            html = row.get("html") or ""
            url = row.get("url") or ""
            if not html:
                continue

            text = extract_text_from_html(html, url)
            if not text:
                continue
            if not is_quality_content(text):
                continue

            h = mmh3.hash128(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            payload = json.dumps({"text": text, "source_url": url}, ensure_ascii=False) + "\n"
            b = len(payload.encode("utf-8"))
            if sample_bytes + b > TARGET_SAMPLE_BYTES:
                break

            out.write(payload)
            sample_bytes += b
            rows_out += 1
            pbar.update(b)

    print(f"Sample: {sample_bytes / (1024**2):.2f} MB, {rows_out} lines -> {SAMPLE_PATH}")
    if sample_bytes < TARGET_SAMPLE_BYTES * 0.85:
        print(
            "Warning: sample is below the target size. The site may have limited unique text after dedupe; "
            "run more raw passes (increase COMMANDO_MAX_PASSES / raw size) or relax filters."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="COMMANDO raw + cleaned dataset pipeline")
    parser.add_argument("--raw-only", action="store_true", help="Only build raw_commando_crawl.jsonl")
    parser.add_argument("--clean-only", action="store_true", help="Only build commando_networks.jsonl from raw")
    args = parser.parse_args()

    if args.raw_only and args.clean_only:
        parser.error("Use only one of --raw-only / --clean-only")

    if not args.clean_only:
        urls = discover_urls()
        print(f"Discovered {len(urls)} URLs (cap {DISCOVERY_CAP}).")
        build_raw_corpus(urls)

    if not args.raw_only:
        clean_to_sample()


if __name__ == "__main__":
    main()
