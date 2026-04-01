from __future__ import annotations

from pathlib import Path
import subprocess


def _run_qwen_caption(
    llama_cli_path: Path,
    model_path: Path,
    mmproj_path: Path,
    image_path: Path,
    prompt: str,
) -> str:
    cmd = [
        str(llama_cli_path),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--image",
        str(image_path),
        "-p",
        prompt,
        "-n",
        "220",
        "--temp",
        "0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "qwen caption command failed")
    out = (res.stdout or "").strip()
    if not out:
        raise RuntimeError("empty qwen caption output")
    return out.splitlines()[-1].strip()


def caption_folder_with_qwen_gguf(
    dataset_dir: Path,
    trigger_word: str,
    llama_cli_path: Path,
    model_path: Path,
    mmproj_path: Path,
) -> dict:
    """
    Caption PNG dataset files using Qwen3-VL GGUF through llama.cpp CLI.
    """
    prompt = (
        "Describe this image in detail for use as a LoRA training caption. "
        "Include appearance, clothing, pose, expression, setting, and lighting. "
        "Return only the caption text."
    )
    images = sorted(dataset_dir.glob("*.png"))
    failed = 0
    short = 0

    for img in images:
        txt = img.with_suffix(".txt")
        try:
            cap = _run_qwen_caption(llama_cli_path, model_path, mmproj_path, img, prompt)
            cap = " ".join(cap.split()).strip(" ,")
            if not cap:
                raise RuntimeError("empty caption")
            txt.write_text(f"{trigger_word}, {cap}", encoding="utf-8")
            if len(cap.split()) < 10:
                short += 1
        except Exception:
            failed += 1
            txt.write_text(f"{trigger_word}, person", encoding="utf-8")

    return {"total": len(images), "failed": failed, "short": short}

