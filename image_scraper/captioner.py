"""
captioner.py — Local image captioner using JoyCaption (fancyfeast/llama-joycaption-beta-one-hf-llava).

Runs fully locally — no API key required.
Model is downloaded from HuggingFace on first use (~17 GB, cached afterwards).

VRAM requirements:
    4-bit (default) : ~5-6 GB  — fits an 8 GB GPU
    8-bit           : ~10 GB
    bfloat16 (full) : ~17 GB   — requires 24 GB+ GPU

For every image a .txt sidecar is written beside it:
    img_001.jpg  ->  img_001.txt

Two public functions:
    caption_folder(folder, ...)  — non-recursive, used after scraping (Mode 1)
    caption_tree(root, ...)      — recursive, used on existing folders (Mode 2)

Usage (standalone):
    python captioner.py images/Dogs              # non-recursive
    python captioner.py images/Dogs --recursive  # walk all subfolders
    python captioner.py images/Dogs --quantize 8bit
    python captioner.py images/Dogs --quantize none   # full bfloat16
    python captioner.py images/Dogs --dry-run
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

CAPTION_PROMPT = "Write a descriptive caption for this image in a formal tone."


@dataclass
class CaptionResult:
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    failed_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model loading (done once, reused across all images)
# ---------------------------------------------------------------------------

def _load_model(quantize: str = "4bit"):
    """
    Load the JoyCaption processor and model.

    Args:
        quantize: "4bit" (default), "8bit", or "none" (full bfloat16).
    """
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install them with:\n"
            "  pip install torch transformers accelerate bitsandbytes"
        ) from exc

    print(f"  Loading JoyCaption ({quantize} precision) — this may take a moment...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        # bitsandbytes quantizes the vision tower even though it shouldn't.
        # Fix it by replacing all quantized linear layers inside the vision
        # tower and projector with dequantized bfloat16 equivalents.
        _dequantize_vision_components(model)
    elif quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
        _dequantize_vision_components(model)
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return processor, model


def _dequantize_vision_components(model) -> None:
    """
    Replace any bitsandbytes-quantized linear layers inside the vision tower
    and multi-modal projector with regular bfloat16 Linear layers.

    LlavaForConditionalGeneration wraps its internals inside a .model attribute
    (LlavaModel), so vision_tower / multi_modal_projector live at model.model.*,
    not model.* directly.  We check both levels to be safe.

    bitsandbytes quantizes these components even though they should stay in
    bfloat16, causing a BFloat16 vs Byte dtype mismatch at inference time.
    """
    import torch
    import bitsandbytes as bnb
    import torch.nn as nn

    def _replace_quantized(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                w = bnb.functional.dequantize_4bit(
                    child.weight.data, child.weight.quant_state
                ).to(torch.bfloat16)
                new = nn.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=w.device, dtype=torch.bfloat16,
                )
                new.weight = nn.Parameter(w)
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.to(torch.bfloat16))
                setattr(parent, name, new)
            elif isinstance(child, bnb.nn.Linear8bitLt):
                w = child.weight.data.to(torch.bfloat16)
                new = nn.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=w.device, dtype=torch.bfloat16,
                )
                new.weight = nn.Parameter(w)
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.to(torch.bfloat16))
                setattr(parent, name, new)
            else:
                _replace_quantized(child)

    # LlavaForConditionalGeneration nests its sub-models inside .model
    inner = getattr(model, "model", model)

    replaced = 0
    for attr in ("vision_tower", "multi_modal_projector"):
        component = getattr(inner, attr, None) or getattr(model, attr, None)
        if component is not None:
            _replace_quantized(component)
            # Keep on GPU — device_map may have placed them there as 4-bit;
            # after dequantization we must pin them to CUDA explicitly.
            if torch.cuda.is_available():
                component.to(dtype=torch.bfloat16, device="cuda")
            else:
                component.to(torch.bfloat16)
            replaced += 1

    if replaced == 0:
        print("  Warning: vision_tower / multi_modal_projector not found — skipping dequantization.")


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def _caption_one(processor, model, img_path: Path) -> str:
    """Run JoyCaption inference on one image and return the caption string."""
    import torch
    from PIL import Image

    image = Image.open(img_path).convert("RGB")

    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user",   "content": CAPTION_PROMPT},
    ]
    convo_string = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[convo_string], images=[image], return_tensors="pt"
    ).to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )[0]

    # Strip the prompt tokens from the output.
    output_ids = output_ids[inputs["input_ids"].shape[1]:]
    caption = processor.tokenizer.decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return caption.strip()


# ---------------------------------------------------------------------------
# Public API — called by main.py
# ---------------------------------------------------------------------------

def _run_captioning(
    images: list[Path],
    quantize: str,
    dry_run: bool,
    progress_callback,
) -> CaptionResult:
    """
    Core loop: caption a pre-built list of image paths.
    Model is loaded lazily on the first image that actually needs it.
    """
    result = CaptionResult()
    processor = model = None

    for img_path in images:
        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists():
            result.skipped += 1
            if progress_callback:
                progress_callback(img_path, None, skipped=True)
            continue

        if dry_run:
            result.generated += 1
            if progress_callback:
                progress_callback(img_path, "[dry run]", skipped=False)
            continue

        if model is None:
            processor, model = _load_model(quantize=quantize)

        try:
            caption = _caption_one(processor, model, img_path)
            txt_path.write_text(caption, encoding="utf-8")
            result.generated += 1
            success = True
        except Exception as exc:  # noqa: BLE001
            import traceback
            if result.failed == 0:  # Print full trace only for the first failure.
                traceback.print_exc()
            caption = str(exc)
            result.failed += 1
            result.failed_files.append(img_path.name)
            success = False

        if progress_callback:
            progress_callback(img_path, caption, skipped=False, success=success)

    return result


def caption_folder(
    folder: Path,
    dry_run: bool = False,
    quantize: str = "4bit",
    progress_callback=None,
) -> CaptionResult:
    """
    Caption every supported image directly inside *folder* (non-recursive).

    Used by Mode 1 (auto, after scraping). Subfolders are ignored.

    Args:
        folder:            Directory to scan (one level only).
        dry_run:           Report what would be captioned without inference.
        quantize:          "4bit" (default), "8bit", or "none".
        progress_callback: Optional callable(path, caption, skipped, success).

    Returns:
        CaptionResult with generated / skipped / failed counts.
    """
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
    return _run_captioning(images, quantize, dry_run, progress_callback)


def caption_tree(
    root: Path,
    dry_run: bool = False,
    quantize: str = "4bit",
    progress_callback=None,
) -> CaptionResult:
    """
    Caption every supported image anywhere inside *root* (recursive).

    Used by Mode 2 (manual, existing folder). Walks all subfolders.
    Images that already have a .txt sidecar are skipped automatically.

    Args:
        root:              Root directory to walk recursively.
        dry_run:           Report what would be captioned without inference.
        quantize:          "4bit" (default), "8bit", or "none".
        progress_callback: Optional callable(path, caption, skipped, success).

    Returns:
        CaptionResult with generated / skipped / failed counts.
    """
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    return _run_captioning(images, quantize, dry_run, progress_callback)


def find_images(root: Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of supported image paths under *root*."""
    if recursive:
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in SUPPORTED)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JoyCaption captions for every image in a folder."
    )
    parser.add_argument("folder", type=Path, help="Folder containing images")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Walk all subfolders recursively",
    )
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Model precision: 4bit (~6 GB VRAM), 8bit (~10 GB), none (~17 GB) [default: 4bit]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be captioned without running the model",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a valid directory.")
        raise SystemExit(1)

    tag = " [DRY RUN]" if args.dry_run else ""
    mode = "recursively" if args.recursive else "non-recursively"
    print(f"Captioning {mode}{tag}: {args.folder.resolve()}")

    fn = caption_tree if args.recursive else caption_folder
    try:
        result = fn(args.folder, dry_run=args.dry_run, quantize=args.quantize)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"  Generated : {result.generated}")
    if result.skipped:
        print(f"  Skipped   : {result.skipped} (already captioned)")
    if result.failed:
        print(f"  Failed    : {result.failed}")
        for name in result.failed_files:
            print(f"    - {name}")


if __name__ == "__main__":
    main()
