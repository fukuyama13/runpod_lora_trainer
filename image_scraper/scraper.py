"""
scraper.py — Bing Images scraper using requests (no API key required).

Bing embeds full-resolution image URLs as JSON metadata inside its HTML,
accessible via a simple regex — no JavaScript execution needed.

Usage (standalone):
    python scraper.py --query "Dogs" --limit 20
"""

import argparse
import re
import time
import urllib.parse
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")

# Bing returns up to ~35 results per page; we paginate using `first`.
_BING_PAGE_SIZE = 35


def safe_folder_name(name: str) -> str:
    """Convert a string to a safe folder name, preserving original case."""
    safe = re.sub(r"[^\w\s-]", "", name)
    return re.sub(r"\s+", "_", safe).strip("_")


def _extract_urls_from_html(html: str) -> list[str]:
    """
    Pull original image URLs from Bing's embedded JSON metadata.

    Bing HTML-encodes its JSON, so `"` becomes `&quot;` and the
    `murl` field holds the direct link to the source image.
    """
    # HTML-encoded variant (most common in modern Bing pages).
    urls = re.findall(r'murl&quot;:&quot;(https?://[^&]+?)&quot;', html)

    # Plain JSON variant (some Bing A/B test variants).
    if not urls:
        urls = re.findall(r'"murl"\s*:\s*"(https?://[^"]+?)"', html)

    return urls


def fetch_image_urls(query: str, limit: int = 20) -> list[str]:
    """
    Search Bing Images and return a list of direct image URLs.

    Args:
        query: Search string.
        limit: Maximum number of unique URLs to return.

    Paginates automatically when `limit` exceeds one page of results.
    """
    seen: set[str] = set()
    unique: list[str] = []
    first = 1

    while len(unique) < limit:
        params = {
            "q": query,
            "count": _BING_PAGE_SIZE,
            "first": first,
            "safeSearch": "Off",
        }
        url = "https://www.bing.com/images/search?" + urllib.parse.urlencode(params)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch Bing Images page: {exc}") from exc

        page_urls = _extract_urls_from_html(resp.text)

        if not page_urls:
            break  # No more results available.

        for u in page_urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
                if len(unique) >= limit:
                    break

        first += _BING_PAGE_SIZE
        time.sleep(0.5)  # Polite pause between pages.

    return unique[:limit]


def _extension_from_url(url: str) -> str:
    """Guess file extension from a URL path, defaulting to .jpg."""
    path = urllib.parse.urlparse(url).path.lower()
    for ext in IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ".jpg"


def download_images(
    urls: list[str],
    term: str,
    output_root: str = "images",
    progress_callback=None,
) -> tuple[int, Path]:
    """
    Download each URL and save to images/<term>/img_NNN.<ext>.

    Args:
        urls:              Direct image URLs to download.
        term:              Search subject — used for the subfolder name.
        output_root:       Root directory for all downloads (default: images/).
        progress_callback: Optional callable(url, success, dest_path) fired
                           after each attempt; used by main.py for tqdm.

    Returns:
        (number_successfully_downloaded, destination_folder_path)
    """
    folder = Path(output_root) / safe_folder_name(term)
    folder.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for idx, url in enumerate(urls, start=1):
        ext = _extension_from_url(url)
        dest = folder / f"img_{idx:03d}{ext}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"Unexpected Content-Type: {content_type!r}")

            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)

            downloaded += 1
            success = True

        except Exception:  # noqa: BLE001
            dest = None
            success = False

        if progress_callback:
            progress_callback(url, success, dest)

        time.sleep(0.2)  # Polite pause between downloads.

    return downloaded, folder


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download images from Bing Images without an API key."
    )
    parser.add_argument("--query", required=True, help="Search term")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of images to download (default: 20)",
    )
    parser.add_argument(
        "--output",
        default="images",
        help="Root output directory (default: images/)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Searching Bing Images for: '{args.query}'")
    urls = fetch_image_urls(args.query, limit=args.limit)
    print(f"Found {len(urls)} image URL(s). Downloading...\n")

    def _cb(url: str, success: bool, dest) -> None:
        print(f"  saved -> {dest}" if success else "  SKIP (download failed)")

    count, folder = download_images(
        urls,
        term=args.query,
        output_root=args.output,
        progress_callback=_cb,
    )

    print(f"\nDone. {count}/{len(urls)} images saved to: {folder}")


if __name__ == "__main__":
    main()
