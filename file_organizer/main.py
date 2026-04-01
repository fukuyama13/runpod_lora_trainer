from __future__ import annotations

from pathlib import Path
import sys

from mover import move_files
from scanner import REPORT_ORDER, scan_files


def _supports_block_char() -> bool:
    encoding = sys.stdout.encoding or ""
    try:
        "█".encode(encoding)
        return True
    except Exception:
        return False


BAR_CHAR = "█" if _supports_block_char() else "#"
EMPTY_CHAR = "░" if _supports_block_char() else "-"
FULL_BAR = BAR_CHAR * 20


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_size(size_bytes: int) -> str:
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.1f} MB"


def _print_report(stats: dict[str, dict[str, int]], destination: str) -> tuple[int, int]:
    total_count = sum(stats[category]["count"] for category in REPORT_ORDER)
    total_size = sum(stats[category]["size_bytes"] for category in REPORT_ORDER)

    print("\n============================================")
    print("REPORT")
    print("============================================")

    for category in REPORT_ORDER:
        count = stats[category]["count"]
        size = _format_size(stats[category]["size_bytes"])
        print(f"{category:<11} {_format_count(count):>7} files   ({size})")

    print("--------------------------------------------")
    print(f"Total:     {_format_count(total_count):>7} files  ({_format_size(total_size)})\n")
    print(f"Destination: {destination}")
    print("============================================\n")
    return total_count, total_size


def _render_progress(current: int, total: int) -> None:
    if total <= 0:
        return
    width = 30
    filled = int((current / total) * width)
    bar = (BAR_CHAR * filled).ljust(width, EMPTY_CHAR)
    print(f"\rMoving files... {bar} {current}/{total}", end="", flush=True)
    if current == total:
        print()


def main() -> None:
    source = input("Drive/folder to scan: ").strip().strip('"')
    destination = input("Destination folder: ").strip().strip('"')

    source_path = Path(source).expanduser()
    if not source_path.exists():
        print("Source does not exist. Exiting with no changes made.")
        return

    print("\nScanning... please wait")
    files, stats = scan_files(str(source_path))
    print(f"{FULL_BAR} {_format_count(len(files))} files found")

    _print_report(stats, destination)

    confirm = input("Do you want to proceed? [y/n]: ").strip().lower()
    if confirm != "y":
        print("No changes made.")
        return

    destination_path = Path(destination).expanduser()
    destination_path.mkdir(parents=True, exist_ok=True)
    log_path = destination_path / "organized_log.txt"

    result = move_files(
        files=files,
        destination_root=str(destination_path),
        log_path=str(log_path),
        progress_callback=_render_progress,
    )

    print(f"Done. Moved {result['moved']} of {result['total']} files. Failed: {result['failed']}.")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
