from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys
import traceback


TRAINING_NAME = "Irina"
TRIGGER_WORD = "ohwx_irina"
DATASET_DIR = Path(r"C:\lora_projects\Irina\dataset\20_ohwx_irina")
LOG_PATH = Path(r"C:\lora_projects\Irina\logs\agent.log")

ARTIFACT_RE = re.compile(
    r"\b(watermark|compression noise|jpeg artifacts?|artifacting|logo)\b", re.IGNORECASE
)


def log_line(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def clean_caption(caption: str) -> str:
    caption = ARTIFACT_RE.sub("", caption)
    return " ".join(caption.split()).strip(" ,")


def main() -> int:
    # Reuse the exact JoyCaption approach from image_scraper/captioner.py.
    sys.path.insert(0, str(Path(__file__).parent / "image_scraper"))
    from captioner import _caption_one, _load_model  # type: ignore

    images = sorted(DATASET_DIR.glob("*.png"))
    log_line(
        f"STEP3 captioning started | training={TRAINING_NAME} | captioner=joycaption(reuse:image_scraper) | images={len(images)}"
    )

    if not images:
        print("ERROR: no images found in dataset folder")
        log_line("STEP3 captioning aborted: no images found")
        return 1

    try:
        processor, model = _load_model(quantize="4bit")
    except Exception as exc:
        print(f"ERROR loading JoyCaption model: {exc}")
        log_line(f"JoyCaption load failed: {exc}")
        return 1

    short_count = 0
    fallback_count = 0
    samples: list[tuple[str, str]] = []

    for idx, img_path in enumerate(images, start=1):
        txt_path = img_path.with_suffix(".txt")
        try:
            generated = _caption_one(processor, model, img_path)
            generated = clean_caption(generated)
            if not generated:
                raise RuntimeError("empty caption generated")

            final_caption = f"{TRIGGER_WORD}, {generated}"
            txt_path.write_text(final_caption, encoding="utf-8")

            if len(generated.split()) < 10:
                short_count += 1
                log_line(f"Caption under 10 words flagged: {txt_path.name}")

            if len(samples) < 3:
                samples.append((txt_path.name, final_caption))

            log_line(f"Captioned {img_path.name} -> {txt_path.name} ({idx}/{len(images)})")
        except Exception as exc:
            fallback_count += 1
            txt_path.write_text(f"{TRIGGER_WORD}, person", encoding="utf-8")
            log_line(f"Caption failed for {img_path.name}; fallback used. Error: {exc}")
            log_line(traceback.format_exc().strip())

    log_line(
        "STEP3 captioning completed | "
        f"total={len(images)} | short={short_count} | fallback={fallback_count} | quantize=4bit"
    )

    print("Captioner: JoyCaption (reused image_scraper approach, 4bit)")
    print(f"Total images: {len(images)}")
    print(f"Short captions (<10 words): {short_count}")
    print(f"Fallback captions used: {fallback_count}")
    print("Samples:")
    for name, cap in samples:
        print(f"  {name} -> {cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
