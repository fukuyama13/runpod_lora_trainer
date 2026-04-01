from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _default_projects_root() -> Path:
    # On RunPod/Linux use /workspace by default, keep Windows local default.
    if os.name == "nt":
        return Path(r"C:\lora_projects")
    return Path("/workspace/lora_projects")


PROJECTS_ROOT = Path(os.environ.get("LORA_PROJECTS_ROOT", str(_default_projects_root())))


def ask(prompt: str) -> str:
    return input(prompt).strip()


def must(prompt: str) -> str:
    while True:
        v = ask(prompt)
        if v:
            return v


def maybe(prompt: str, default: str | None = None) -> str | None:
    v = ask(prompt)
    if v:
        return v
    return default


def trigger(name: str) -> str:
    return "ohwx_" + name.lower().replace(" ", "_")


def init_project(name: str, trig: str) -> dict[str, Path]:
    base = PROJECTS_ROOT / name
    paths = {
        "base": base,
        "input": base / "input",
        "dataset": base / "dataset" / f"20_{trig}",
        "config": base / "config",
        "output": base / "output",
        "samples": base / "samples",
        "logs": base / "logs",
        "scrape_tmp": base / "_scrape_tmp",
        "session": base / "logs" / "session.json",
    }
    for k in ("input", "dataset", "config", "output", "samples", "logs"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def choose_captioner(default: str | None = None) -> tuple[str, dict]:
    if default == "joycaption":
        return "joycaption", {"quantize": "4bit"}
    if default == "qwen3_vl_gguf":
        return "qwen3_vl_gguf", {
            "llama_cli_path": must("Path to llama-cli.exe: "),
            "model_path": must("Path to Qwen GGUF model file: "),
            "mmproj_path": must("Path to Qwen mmproj GGUF file: "),
        }

    print("\nCaption backend:")
    print("  1) joycaption")
    print("  2) qwen3_vl_gguf")
    while True:
        c = ask("Choose [1/2]: ")
        if c == "1":
            return "joycaption", {"quantize": "4bit"}
        if c == "2":
            llama_cli = must("Path to llama-cli.exe: ")
            model = must("Path to Qwen GGUF model file: ")
            mmproj = must("Path to Qwen mmproj GGUF file: ")
            return "qwen3_vl_gguf", {
                "llama_cli_path": llama_cli,
                "model_path": model,
                "mmproj_path": mmproj,
            }


def scrape_to_temp(paths: dict[str, Path]) -> Path | None:
    # Reuse existing image_scraper implementation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_scraper.cleaner import clean_folder
    from image_scraper.scraper import download_images, fetch_image_urls

    term = must("Search term: ")
    raw_limit = ask("How many images? [40]: ")
    raw_min = ask("Min size px [512, 0 to skip]: ")
    limit = int(raw_limit) if raw_limit.isdigit() and int(raw_limit) > 0 else 40
    min_size = int(raw_min) if raw_min.isdigit() else 512

    urls = fetch_image_urls(term, limit=limit)
    if not urls:
        print("No results found.")
        return None
    downloaded, folder = download_images(urls, term=term, output_root=str(paths["scrape_tmp"]))
    if downloaded == 0:
        print("No images downloaded.")
        return None
    if min_size > 0:
        clean_folder(folder, min_size=min_size, dry_run=False)
    return folder


def launch_review(source: Path, input_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    script = Path(__file__).parent / "review_queue.py"
    cmd = [
        sys.executable,
        str(script),
        "--source",
        str(source),
        "--input",
        str(input_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(cmd, **kwargs)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Review UI: http://{display_host}:{port}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA Fresh Pipeline starter")
    ap.add_argument("--training-name", default=None)
    ap.add_argument("--captioner", choices=["joycaption", "qwen3_vl_gguf"], default=None)
    ap.add_argument("--source-mode", choices=["manual", "web"], default=None)
    ap.add_argument("--review-host", default=os.environ.get("REVIEW_HOST", "127.0.0.1"))
    ap.add_argument("--review-port", type=int, default=8766)
    ap.add_argument("--runpod", action="store_true", help="Use RunPod-friendly defaults")
    ap.add_argument("--save-env-tokens", action="store_true", help="Store HF/CivitAI token refs in session")
    return ap.parse_args()


def main() -> int:
    global PROJECTS_ROOT
    args = parse_args()
    if args.runpod:
        os.environ.setdefault("LORA_PROJECTS_ROOT", "/workspace/lora_projects")
        PROJECTS_ROOT = Path(os.environ["LORA_PROJECTS_ROOT"])
        if args.review_host == "127.0.0.1":
            args.review_host = "0.0.0.0"

    print("=" * 54)
    print("LoRA Fresh Pipeline")
    print("=" * 54)

    name = args.training_name or must("Training name: ")
    trig = trigger(name)
    captioner, captioner_cfg = choose_captioner(default=args.captioner)
    paths = init_project(name, trig)

    session = {
        "training_name": name,
        "trigger_word": trig,
        "captioner": captioner,
        "captioner_config": captioner_cfg,
        "projects_root": str(PROJECTS_ROOT),
        "runpod_mode": bool(args.runpod),
    }
    if args.save_env_tokens:
        session["tokens"] = {
            "hf_token_present": bool(os.environ.get("HF_TOKEN")),
            "civitai_token_present": bool(os.environ.get("CIVITAI_API_TOKEN")),
        }
    paths["session"].write_text(json.dumps(session, indent=2), encoding="utf-8")

    mode = args.source_mode
    if mode is None:
        print("\nImage source:")
        print("  1) Manual add")
        print("  2) Web scrape + review")
        mode = "web" if ask("Choose [1/2]: ") == "2" else "manual"

    if mode == "web":
        src = scrape_to_temp(paths)
        if src:
            launch_review(src, paths["input"], host=args.review_host, port=args.review_port)
            print("Review and move images to input, then continue with validation/preprocess.")
    else:
        print(f"Add images to: {paths['input']}")

    print("\nProject ready.")
    print(f"Base       : {paths['base']}")
    print(f"Trigger    : {trig}")
    print(f"Captioner  : {captioner}")
    print(f"Session    : {paths['session']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

