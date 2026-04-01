from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


FILE_ATTRIBUTE_HIDDEN = 0x2
FILE_ATTRIBUTE_SYSTEM = 0x4


CATEGORY_EXTENSIONS: Dict[str, set[str]] = {
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic", ".svg"},
    "Videos": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpeg", ".mpg"},
    "Music": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"},
    "Documents": {
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".rtf",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".csv",
        ".epub",
    },
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Code": {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".go",
        ".rb",
        ".php",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".yml",
        ".yaml",
        ".sql",
        ".sh",
        ".ps1",
    },
}


REPORT_ORDER = ["Images", "Videos", "Music", "Documents", "Archives", "Code", "Other"]


@dataclass
class FileEntry:
    source_path: Path
    category: str
    size_bytes: int


def _is_hidden_or_system(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True

    try:
        stat_result = path.stat()
        attrs = getattr(stat_result, "st_file_attributes", 0)
        return bool(attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM))
    except OSError:
        return True


def _category_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    for category, ext_set in CATEGORY_EXTENSIONS.items():
        if suffix in ext_set:
            return category
    return "Other"


def scan_files(source_root: str) -> Tuple[List[FileEntry], Dict[str, Dict[str, int]]]:
    source_path = Path(source_root).expanduser()
    entries: List[FileEntry] = []
    stats = {category: {"count": 0, "size_bytes": 0} for category in REPORT_ORDER}

    for current_root, dir_names, file_names in os.walk(source_path):
        current_dir = Path(current_root)

        # Remove hidden/system directories from traversal.
        filtered_dir_names = []
        for dir_name in dir_names:
            dir_path = current_dir / dir_name
            if not _is_hidden_or_system(dir_path):
                filtered_dir_names.append(dir_name)
        dir_names[:] = filtered_dir_names

        for file_name in file_names:
            file_path = current_dir / file_name
            if _is_hidden_or_system(file_path):
                continue

            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                continue

            category = _category_for_path(file_path)
            entries.append(FileEntry(source_path=file_path, category=category, size_bytes=size_bytes))
            stats[category]["count"] += 1
            stats[category]["size_bytes"] += size_bytes

    return entries, stats
