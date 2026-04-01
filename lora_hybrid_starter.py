"""
Unified LoRA project starter.

Supports two image source modes:
1) Manual: user places images in input folder
2) Web scrape: reuses image_scraper modules to fetch images from Bing

Creates LoRA project structure under C:\lora_projects\<training_name>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path


PROJECTS_ROOT = Path(r"C:\lora_projects")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _ask(prompt: str) -> str:
    return input(prompt).strip()


def _must_input(prompt: str) -> str:
    while True:
        value = _ask(prompt)
        if value:
            return value
        print("Please enter a value.")


def _choose_captioner() -> str:
    print("\nWhich captioning model would you like to use?")
    print("  1. Florence-2")
    print("  2. Florence-2 ORT")
    print("  3. JoyCaption")
    while True:
        choice = _ask("Type 1, 2, or 3: ")
        if choice == "1":
            return "florence2"
        if choice == "2":
            return "florence2_ort"
        if choice == "3":
            return "joycaption"
        print("Invalid choice. Please type 1, 2, or 3.")


def _trigger_word(training_name: str) -> str:
    return "ohwx_" + training_name.lower().replace(" ", "_")


def _project_paths(training_name: str, trigger: str) -> dict[str, Path]:
    base = PROJECTS_ROOT / training_name
    return {
        "base": base,
        "input": base / "input",
        "dataset": base / "dataset" / f"20_{trigger}",
        "config": base / "config",
        "output": base / "output",
        "samples": base / "samples",
        "logs": base / "logs",
        "session": base / "logs" / "session.json",
        "scrape_tmp": base / "_scrape_tmp",
    }


def _ensure_paths(paths: dict[str, Path]) -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    for key in ("input", "dataset", "config", "output", "samples", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _print_tree(training_name: str, trigger: str) -> None:
    print("\nProject structure:")
    print(f"{PROJECTS_ROOT}\\")
    print(f"\\-- {training_name}\\")
    print("    |-- input\\")
    print("    |-- dataset\\")
    print(f"    |   \\-- 20_{trigger}\\")
    print("    |-- config\\")
    print("    |-- output\\")
    print("    |-- samples\\")
    print("    \\-- logs\\")


def _save_session(paths: dict[str, Path], training_name: str, trigger: str, captioner: str) -> None:
    session = {
        "training_name": training_name,
        "trigger_word": trigger,
        "captioner": captioner,
    }
    paths["session"].write_text(json.dumps(session, indent=2), encoding="utf-8")


def _copy_supported_images(src: Path, dst: Path) -> int:
    copied = 0
    for file in sorted(src.iterdir()):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS:
            out = dst / file.name
            if out.exists():
                stem = file.stem
                ext = file.suffix
                i = 2
                while True:
                    candidate = dst / f"{stem}_{i}{ext}"
                    if not candidate.exists():
                        out = candidate
                        break
                    i += 1
            shutil.copy2(file, out)
            copied += 1
    return copied


def _launch_review_ui(source_folder: Path, input_dir: Path, port: int) -> None:
    review_script = Path(__file__).parent / "review_scraped_images.py"
    cmd = [
        sys.executable,
        str(review_script),
        "--source",
        str(source_folder),
        "--input",
        str(input_dir),
        "--port",
        str(port),
    ]
    # Start review server detached from this process.
    subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))


def _run_web_scrape(
    paths: dict[str, Path],
    search_term: str | None,
    limit: int | None,
    min_size: int | None,
    launch_review: bool,
    review_port: int,
) -> None:
    sys.path.insert(0, str(Path(__file__).parent / "image_scraper"))
    from cleaner import clean_folder
    from scraper import download_images, fetch_image_urls

    term = search_term or _must_input("\nSearch term (e.g. Irina actress red carpet): ")
    if limit is None:
        raw_limit = _ask("How many images? [default: 40]: ")
        limit = int(raw_limit) if raw_limit.isdigit() and int(raw_limit) > 0 else 40
    if min_size is None:
        raw_min_size = _ask("Min size cleanup in px? [default: 512, 0 to skip]: ")
        min_size = int(raw_min_size) if raw_min_size.isdigit() else 512

    print(f"\nSearching Bing for '{term}'...")
    urls = fetch_image_urls(term, limit=limit)
    if not urls:
        print("No URLs found.")
        return

    print(f"Found {len(urls)} URL(s). Downloading...")
    downloaded, folder = download_images(
        urls=urls,
        term=term,
        output_root=str(paths["scrape_tmp"]),
    )
    print(f"Downloaded {downloaded} image(s).")

    if downloaded == 0:
        return

    if min_size > 0:
        cleaned = clean_folder(folder, min_size=min_size, dry_run=False)
        print(f"Removed {cleaned.removed} low-quality image(s); kept {cleaned.kept}.")

    if launch_review:
        print("\nLaunching review UI so you can move/discard manually...")
        _launch_review_ui(folder, paths["input"], review_port)
        print(f"Review URL: http://127.0.0.1:{review_port}")
        print(f"Source review folder: {folder}")
    else:
        copied = _copy_supported_images(folder, paths["input"])
        print(f"Copied {copied} image(s) into: {paths['input']}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified LoRA starter (manual or web scrape).")
    p.add_argument("--training-name", default=None, help="Training/project name.")
    p.add_argument(
        "--captioner",
        choices=["florence2", "florence2_ort", "joycaption"],
        default=None,
        help="Captioner to store in session metadata.",
    )
    p.add_argument(
        "--source-mode",
        choices=["manual", "web"],
        default=None,
        help="Image source mode.",
    )
    p.add_argument("--search-term", default=None, help="Bing search term for web mode.")
    p.add_argument("--limit", type=int, default=None, help="Image download limit for web mode.")
    p.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Minimum size cleanup for web mode (0 disables cleanup).",
    )
    p.add_argument(
        "--launch-review",
        action="store_true",
        help="After web scrape, launch review UI (move/discard manually).",
    )
    p.add_argument("--review-port", type=int, default=8765, help="Review UI port.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print("=" * 56)
    print("SDXL LoRA Hybrid Starter")
    print("=" * 56)

    training_name = args.training_name or _must_input("Training name (e.g. irina): ")
    captioner = args.captioner or _choose_captioner()
    trigger = _trigger_word(training_name)
    paths = _project_paths(training_name, trigger)

    _ensure_paths(paths)
    _save_session(paths, training_name, trigger, captioner)
    _print_tree(training_name, trigger)

    print(f"\nTrigger word: {trigger}")
    print(f"Captioner: {captioner}")

    if args.source_mode:
        mode = "2" if args.source_mode == "web" else "1"
    else:
        print("\nChoose image source:")
        print("  1. Manual (you add images to input folder)")
        print("  2. Web scrape (auto collect from Bing)")
        mode = _ask("Type 1 or 2: ")

    if mode == "2":
        _run_web_scrape(
            paths=paths,
            search_term=args.search_term,
            limit=args.limit,
            min_size=args.min_size,
            launch_review=args.launch_review,
            review_port=args.review_port,
        )
    else:
        print(f"\nAdd images manually to: {paths['input']}")
        _ask("Press Enter after you finish adding images...")

    image_count = len(
        [
            p
            for p in paths["input"].iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    )
    print(f"\nInput images ready: {image_count}")
    print(f"Next step: continue dataset validation/preprocess from {paths['base']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
