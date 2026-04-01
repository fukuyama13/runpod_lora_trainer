# LoRA Fresh Pipeline

Fresh standalone pipeline that combines:
- Project initialization for LoRA training
- Image source selection (manual or web scrape via existing `image_scraper`)
- Review queue UI (move/discard scraped images)
- Caption backend selection:
  - JoyCaption (existing local backend)
  - Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF

Qwen model page:
- [Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF/tree/main?not-for-all-audiences=true)

## Files

- `pipeline_runner.py` - main starter script
- `review_queue.py` - local web review UI for scraped images
- `caption_backends\joycaption_backend.py` - JoyCaption wrapper
- `caption_backends\qwen3_gguf_backend.py` - Qwen GGUF wrapper (llama.cpp CLI)
- `requirements.txt` - minimal extra deps for this project

## Quick start

```powershell
python "C:\Users\nikos\Desktop\cursor_code\lora_fresh_pipeline\pipeline_runner.py"
```

## RunPod-friendly usage

For RunPod/Jupyter terminals, prefer non-interactive startup and Linux paths:

```bash
export LORA_PROJECTS_ROOT=/workspace/lora_projects
export HF_TOKEN=your_hf_token
export CIVITAI_API_TOKEN=your_civitai_token
python /workspace/lora_fresh_pipeline/pipeline_runner.py \
  --runpod \
  --training-name "Yael Shelbia" \
  --captioner joycaption \
  --source-mode web \
  --review-host 0.0.0.0 \
  --review-port 8766 \
  --save-env-tokens
```

Notes:
- `--runpod` switches defaults for remote Linux execution.
- Use RunPod port mapping to access the review UI (`8766`) and any other service ports.
- CivitAI downloads typically require `CIVITAI_API_TOKEN` in your environment.
- ComfyUI (`8188`) and JupyterLab (`8888`) from your template can run in parallel with this pipeline.

## Qwen GGUF notes

For Qwen GGUF captioning, you need:
- `llama.cpp` binary (e.g. `llama-cli.exe`)
- model GGUF file (e.g. `Q4_K_M`)
- matching `mmproj` GGUF file

The script stores these in session config and uses them when captioning.

