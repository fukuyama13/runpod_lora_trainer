from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from typing import Callable, Iterable

from scanner import FileEntry


ProgressCallback = Callable[[int, int], None]


def _unique_destination_path(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        renamed = target_dir / f"{stem}_{counter}{suffix}"
        if not renamed.exists():
            return renamed
        counter += 1


def move_files(
    files: Iterable[FileEntry],
    destination_root: str,
    log_path: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    destination = Path(destination_root).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path).expanduser()

    file_list = list(files)
    total = len(file_list)
    moved = 0
    failed = 0

    for index, entry in enumerate(file_list, start=1):
        target_dir = destination / entry.category
        target_dir.mkdir(parents=True, exist_ok=True)
        final_destination = _unique_destination_path(target_dir, entry.source_path.name)
        timestamp = datetime.now().isoformat(timespec="seconds")

        try:
            shutil.move(str(entry.source_path), str(final_destination))
            moved += 1
            status = "MOVED"
        except OSError as exc:
            failed += 1
            status = f"FAILED ({exc})"
            final_destination = Path("-")

        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"{timestamp} | {status} | {entry.source_path} -> {final_destination}\n")

        if progress_callback:
            progress_callback(index, total)

    return {"total": total, "moved": moved, "failed": failed}
