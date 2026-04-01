"""
main.py — Interactive CLI entry point for the Bing Images scraper.

Two modes:

  Mode 1 — Scrape and caption
    Search Bing Images, download, optionally clean and caption.
    Saved to: images/<term>/img_001.jpg

  Mode 2 — Caption existing folder
    Point at any folder on disk; captions all images recursively.
    Saves a .txt sidecar next to each image, skips already-captioned ones.

Usage:
    python main.py                            # interactive mode prompt
    python main.py "Dogs"                     # Mode 1, term on CLI
    python main.py "Dogs" --limit 20 --captions
    python main.py --caption-folder "C:/Pictures/holidays"   # Mode 2
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from captioner import caption_folder, caption_tree, find_images
from cleaner import clean_folder
from scraper import download_images, fetch_image_urls, safe_folder_name


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bing image scraper with optional JoyCaption captioning.",
        usage="python main.py [term] [options]  |  python main.py --caption-folder PATH",
    )
    # Mode 1 args
    parser.add_argument(
        "term",
        nargs="?",
        default=None,
        help="(Mode 1) Search term",
    )
    parser.add_argument("--limit",    type=int, default=None, help="Images to download (default: 20)")
    parser.add_argument("--min-size", type=int, default=None, metavar="PX",
                        help="Remove images below PX x PX after download (0 = skip)")
    parser.add_argument("--captions", action=argparse.BooleanOptionalAction, default=None,
                        help="Generate JoyCaption captions (--captions / --no-captions)")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit",
                        help="JoyCaption VRAM mode: 4bit ~6GB, 8bit ~10GB, none ~17GB [default: 4bit]")
    # Mode 2 arg
    parser.add_argument(
        "--caption-folder",
        type=Path,
        default=None,
        metavar="PATH",
        help="(Mode 2) Caption all images in this existing folder recursively",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _input(prompt: str) -> str:
    """Wrap input() to handle Ctrl-C / EOF gracefully."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def prompt_term() -> str:
    term = _input("Search term: ")
    if not term:
        print("Error: search term cannot be empty.")
        sys.exit(1)
    return term


def prompt_limit(default: int = 20) -> int:
    raw = _input(f"How many images? [default: {default}]: ")
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
        return value
    except ValueError:
        print(f"  Invalid number — using default ({default}).")
        return default


def prompt_min_size() -> int:
    raw = _input("Min image size in px? (e.g. 200, or 0 to skip cleaning) [default: 0]: ")
    if not raw:
        return 0
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        print("  Invalid value — skipping cleaning.")
        return 0


def prompt_captions() -> bool:
    raw = _input("Generate captions with JoyCaption? [y/n, default: n]: ").lower()
    return raw in ("y", "yes")


def prompt_mode() -> int:
    """Ask the user which mode to run."""
    print("  What would you like to do?")
    print("  1. Scrape and caption images")
    print("  2. Caption images in an existing folder")
    while True:
        raw = _input("  Choose [1/2]: ")
        if raw in ("1", "2"):
            return int(raw)
        print("  Please enter 1 or 2.")


# ---------------------------------------------------------------------------
# Mode 1 — Scrape (+ optional clean + optional caption)
# ---------------------------------------------------------------------------

def run_mode_1(args: argparse.Namespace) -> None:
    term     = args.term     if args.term     is not None else prompt_term()
    limit    = args.limit    if args.limit    is not None else prompt_limit()
    min_size = args.min_size if args.min_size is not None else prompt_min_size()
    captions = args.captions if args.captions is not None else prompt_captions()
    quantize = args.quantize

    print(f"\n  Term     : {term}")
    print(f"  Limit    : {limit} images")
    if min_size:
        print(f"  Min size : {min_size}px")
    print(f"  Captions : {'yes' if captions else 'no'}")
    print()

    # -- Fetch URLs ----------------------------------------------------------
    print(f"  Searching: {term} ...", end="", flush=True)
    try:
        urls = fetch_image_urls(term, limit=limit)
    except RuntimeError as exc:
        print(f" ERROR: {exc}")
        sys.exit(1)

    if not urls:
        print(" no results found.")
        sys.exit(1)
    print(f" {len(urls)} URL(s) found")

    # -- Download ------------------------------------------------------------
    bar = tqdm(total=len(urls), unit="img", ncols=60, leave=False,
               bar_format="    {bar}| {n_fmt}/{total_fmt}")

    def on_image(_url, success, _dest):
        bar.update(1)

    downloaded, folder = download_images(urls, term=term, output_root="images",
                                         progress_callback=on_image)
    bar.close()

    failed = len(urls) - downloaded
    if failed:
        print(f"  Downloaded : {downloaded} saved  ({failed} failed)")
    else:
        print(f"  Downloaded : {downloaded} images saved")

    # -- Clean ---------------------------------------------------------------
    if min_size and downloaded > 0:
        try:
            cleaned = clean_folder(folder, min_size=min_size)
            downloaded -= cleaned.removed
            if cleaned.removed:
                print(f"  Cleaned    : removed {cleaned.removed} image(s) "
                      f"below {min_size}px  ({cleaned.kept} kept)")
        except RuntimeError as exc:
            print(f"  Cleaner    : {exc}")

    # -- Caption -------------------------------------------------------------
    if captions and downloaded > 0:
        print(f"  Captioning : {safe_folder_name(term)}...", end="", flush=True)
        try:
            cap_bar = tqdm(total=None, unit="img", ncols=50, leave=False,
                           bar_format="  {n_fmt} captioned")

            def on_caption(_path, _text, skipped=False, success=True):
                if not skipped:
                    cap_bar.update(1)

            cap_result = caption_folder(folder, quantize=quantize,
                                        progress_callback=on_caption)
            cap_bar.close()

            if cap_result.failed:
                print(f" {cap_result.generated} captions generated  "
                      f"({cap_result.failed} failed)")
            else:
                print(f" {cap_result.generated} captions generated")
        except RuntimeError as exc:
            print(f" ERROR: {exc}")

    # -- Summary -------------------------------------------------------------
    print()
    print("=" * 52)
    label = "kept after cleaning" if min_size else "saved"
    print(f"  Total    : {downloaded} image(s) {label}")
    if failed:
        print(f"  Skipped  : {failed} (failed or invalid)")
    print(f"  Saved to : {folder.resolve()}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Mode 2 — Caption an existing folder (recursive)
# ---------------------------------------------------------------------------

def run_mode_2(args: argparse.Namespace) -> None:
    # Resolve folder — from CLI flag or interactive prompt.
    if args.caption_folder:
        folder = args.caption_folder
    else:
        raw = _input("Folder path: ")
        folder = Path(raw)

    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    quantize = args.quantize

    # Scan recursively so the user sees the count before anything starts.
    images = find_images(folder, recursive=True)
    total  = len(images)

    print(f"\n  Found {total} image(s) in folder")
    if total == 0:
        print("  Nothing to do.")
        sys.exit(0)

    already = sum(1 for p in images if p.with_suffix(".txt").exists())
    to_do   = total - already
    if already:
        print(f"  ({already} already captioned — will skip)")
    print()

    bar = tqdm(total=total, unit="img", ncols=60,
               bar_format="  Captioning |{bar}| {n_fmt}/{total_fmt}")

    def on_caption(_path, _text, skipped=False, success=True):
        bar.update(1)

    try:
        result = caption_tree(folder, quantize=quantize, progress_callback=on_caption)
    except RuntimeError as exc:
        bar.close()
        print(f"\nError: {exc}")
        sys.exit(1)

    bar.close()

    print()
    print("=" * 52)
    print(f"  Done!    : {result.generated} caption(s) saved")
    if result.skipped:
        print(f"  Skipped  : {result.skipped} (already captioned)")
    if result.failed:
        print(f"  Failed   : {result.failed}")
        for name in result.failed_files:
            print(f"    - {name}")
    print(f"  Folder   : {folder.resolve()}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print("=" * 52)
    print("  Bing Image Scraper + JoyCaption")
    print("=" * 52)
    print()

    # Determine mode:
    #   --caption-folder on CLI  -> always Mode 2
    #   term on CLI              -> always Mode 1
    #   nothing                  -> ask
    if args.caption_folder:
        mode = 2
    elif args.term:
        mode = 1
    else:
        mode = prompt_mode()

    print()

    if mode == 1:
        run_mode_1(args)
    else:
        run_mode_2(args)


if __name__ == "__main__":
    main()
