"""
cleaner.py — Post-download image quality filter.

Scans a folder and removes images that are corrupted or fall below a
minimum resolution threshold.  Designed to run after each year's download
inside main.py, but can also be used standalone.

Usage (standalone):
    python cleaner.py images/Dogs/2019 --min-size 100
    python cleaner.py images/Dogs/2019 --min-size 200 --dry-run
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class CleanResult:
    kept: int = 0
    removed: int = 0
    removed_files: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.kept + self.removed


def clean_folder(
    folder: Path,
    min_size: int = 100,
    dry_run: bool = False,
) -> CleanResult:
    """
    Scan *folder* and remove low-quality or corrupted images.

    An image is removed when ANY of the following are true:
      - The file cannot be opened / decoded (corrupted).
      - Its width  is below *min_size* pixels.
      - Its height is below *min_size* pixels.

    Args:
        folder:   Directory to scan (non-recursive).
        min_size: Minimum acceptable width AND height in pixels (default 100).
        dry_run:  When True, report what would be removed but don't delete.

    Returns:
        CleanResult with kept / removed counts and the list of removed filenames.
    """
    # Import Pillow lazily so the rest of the project doesn't require it
    # unless cleaning is actually used.
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image cleaning. "
            "Install it with: pip install Pillow"
        ) from exc

    result = CleanResult()

    candidates = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED]

    for path in candidates:
        reason = _should_remove(path, min_size, Image, UnidentifiedImageError)
        if reason:
            result.removed += 1
            result.removed_files.append(path.name)
            if not dry_run:
                path.unlink()
        else:
            result.kept += 1

    return result


def _should_remove(path: Path, min_size: int, Image, UnidentifiedImageError) -> str | None:
    """
    Return a reason string if the image should be removed, else None.
    """
    try:
        with Image.open(path) as img:
            w, h = img.size
    except (UnidentifiedImageError, OSError):
        return "corrupted"

    if w < min_size:
        return f"width {w}px < {min_size}px"
    if h < min_size:
        return f"height {h}px < {min_size}px"

    return None


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove corrupted or low-resolution images from a folder."
    )
    parser.add_argument("folder", type=Path, help="Folder to clean")
    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        metavar="PX",
        help="Minimum width AND height in pixels (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without deleting anything",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a valid directory.")
        raise SystemExit(1)

    tag = " [DRY RUN]" if args.dry_run else ""
    print(f"Cleaning{tag}: {args.folder.resolve()}  (min {args.min_size}px)")

    result = clean_folder(args.folder, min_size=args.min_size, dry_run=args.dry_run)

    verb = "Would remove" if args.dry_run else "Removed"
    for name in result.removed_files:
        print(f"  {verb}: {name}")

    print(f"\n  Scanned {result.total} | kept {result.kept} | removed {result.removed}")


if __name__ == "__main__":
    main()
