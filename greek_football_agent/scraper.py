"""
Scrape latest football headlines and article bodies from Greek sports sites.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "el,en-US;q=0.9,en;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(DEFAULT_HEADERS)

# Gazzetta article URLs: /football/<section>/<numeric_id>/<slug>
GAZZETTA_ARTICLE_RE = re.compile(
    r"^https?://(?:www\.)?gazzetta\.gr/football/[^/]+/\d+/[^/?#]+/?$"
)
# Relative paths on listing page
GAZZETTA_REL_ARTICLE_RE = re.compile(r"^/football/[^/]+/\d+/[^/?#]+/?$")


def _fetch(url: str, timeout: int = 30) -> str:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def _clean_text(text: str, max_chars: int = 12000) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    out = "\n".join(ln for ln in lines if ln)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[... truncated ...]"
    return out


def _gazzetta_article_url(href: str, base: str = "https://www.gazzetta.gr") -> str | None:
    if not href:
        return None
    full = urljoin(base, href)
    if "live-" in full or "/live/" in full:
        return None
    if GAZZETTA_ARTICLE_RE.match(full):
        return full.split("?")[0].rstrip("/") + "/"
    if GAZZETTA_REL_ARTICLE_RE.match(href.split("?")[0]):
        return urljoin(base, href.split("?")[0]).rstrip("/") + "/"
    return None


def _sport24_article_url(href: str) -> str | None:
    if not href:
        return None
    full = href if href.startswith("http") else urljoin("https://www.sport24.gr", href)
    p = urlparse(full)
    if p.netloc not in ("www.sport24.gr", "sport24.gr"):
        return None
    path = (p.path or "").split("?")[0]
    if "/football/tag/" in path or "/football/page" in path:
        return None
    if not path.startswith("/football/"):
        return None
    parts = [x for x in path.strip("/").split("/") if x]
    if len(parts) < 2 or parts[0] != "football":
        return None
    if parts[1] in ("tag", "page"):
        return None
    if not path.endswith("/"):
        path = path + "/"
    return f"https://www.sport24.gr{path}"


def fetch_gazzetta_football(limit: int = 5) -> list[dict[str, Any]]:
    base = "https://www.gazzetta.gr/football"
    html = _fetch(base)
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []
    for a in soup.select("a[href]"):
        u = _gazzetta_article_url(a.get("href", ""))
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
        if len(urls) >= limit * 3:
            break

    articles: list[dict[str, Any]] = []
    for url in urls:
        if len(articles) >= limit:
            break
        try:
            art_html = _fetch(url)
            art_soup = BeautifulSoup(art_html, "html.parser")
            title_el = art_soup.select_one("h1") or art_soup.select_one("title")
            title = (title_el.get_text() if title_el else "").strip()
            body_el = art_soup.select_one("article") or art_soup.select_one("main")
            body = _clean_text(body_el.get_text()) if body_el else ""
            if title and body:
                articles.append(
                    {
                        "source": "gazzetta.gr",
                        "title": title,
                        "url": url,
                        "body": body,
                    }
                )
        except requests.RequestException:
            continue
    return articles


def fetch_sport24_football(limit: int = 5) -> list[dict[str, Any]]:
    base = "https://www.sport24.gr/football/"
    html = _fetch(base)
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    urls: list[str] = []
    for a in soup.select("a[href]"):
        u = _sport24_article_url(a.get("href", ""))
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
        if len(urls) >= limit * 4:
            break

    articles: list[dict[str, Any]] = []
    for url in urls:
        if len(articles) >= limit:
            break
        try:
            art_html = _fetch(url)
            art_soup = BeautifulSoup(art_html, "html.parser")
            title_el = art_soup.select_one("h1") or art_soup.select_one("title")
            title = (title_el.get_text() if title_el else "").strip()
            body_el = art_soup.select_one("article")
            body = _clean_text(body_el.get_text()) if body_el else ""
            if title and body:
                articles.append(
                    {
                        "source": "sport24.gr",
                        "title": title,
                        "url": url,
                        "body": body,
                    }
                )
        except requests.RequestException:
            continue
    return articles


def scrape_all(gazzetta_limit: int = 5, sport24_limit: int = 5) -> list[dict[str, Any]]:
    """Return up to `gazzetta_limit` Gazzetta + `sport24_limit` Sport24 articles."""
    out: list[dict[str, Any]] = []
    out.extend(fetch_gazzetta_football(limit=gazzetta_limit))
    out.extend(fetch_sport24_football(limit=sport24_limit))
    return out
