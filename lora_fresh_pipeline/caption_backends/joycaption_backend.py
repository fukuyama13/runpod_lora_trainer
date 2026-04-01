from __future__ import annotations

from pathlib import Path
import re


ARTIFACT_RE = re.compile(
    r"\b(watermark|compression noise|jpeg artifacts?|artifacting|logo)\b", re.IGNORECASE
)


def clean_caption(caption: str) -> str:
    caption = ARTIFACT_RE.sub("", caption)
    return " ".join(caption.split()).strip(" ,")


def caption_folder_with_joycaption(dataset_dir: Path, trigger_word: str, quantize: str = "4bit") -> dict:
    """
    Reuse existing image_scraper JoyCaption backend.
    """
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "image_scraper"))

    from captioner import _caption_one, _load_model  # type: ignore

    images = sorted(dataset_dir.glob("*.png"))
    if not images:
        return {"total": 0, "failed": 0, "short": 0}

    processor, model = _load_model(quantize=quantize)
    failed = 0
    short = 0

    for img in images:
        txt = img.with_suffix(".txt")
        try:
            cap = clean_caption(_caption_one(processor, model, img))
            if not cap:
                raise RuntimeError("empty caption")
            txt.write_text(f"{trigger_word}, {cap}", encoding="utf-8")
            if len(cap.split()) < 10:
                short += 1
        except Exception:
            failed += 1
            txt.write_text(f"{trigger_word}, person", encoding="utf-8")

    return {"total": len(images), "failed": failed, "short": short}

