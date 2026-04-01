from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlparse

from PIL import Image


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, source: Path, input_dir: Path, discard_dir: Path):
        super().__init__(server_address, handler_cls)
        self.source = source
        self.input_dir = input_dir
        self.discard_dir = discard_dir

    def list_images(self) -> list[dict]:
        items: list[dict] = []
        for p in sorted(self.source.iterdir()):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED:
                continue
            w, h = image_dimensions(p)
            items.append(
                {
                    "name": p.name,
                    "width": w,
                    "height": h,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }
            )
        return items

    def unique_target(self, folder: Path, filename: str) -> Path:
        target = folder / filename
        if not target.exists():
            return target
        stem = Path(filename).stem
        ext = Path(filename).suffix
        idx = 2
        while True:
            candidate = folder / f"{stem}_{idx}{ext}"
            if not candidate.exists():
                return candidate
            idx += 1


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, text: str, status: int = 200) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(self._index_html())
            return
        if parsed.path == "/api/images":
            self._json({"images": self.server.list_images()})
            return
        if parsed.path.startswith("/image/"):
            name = parsed.path[len("/image/") :]
            file_path = (self.server.source / name).resolve()
            if not str(file_path).startswith(str(self.server.source.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                data = file_path.read_bytes()
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            ctype = "image/jpeg"
            ext = file_path.suffix.lower()
            if ext == ".png":
                ctype = "image/png"
            elif ext == ".webp":
                ctype = "image/webp"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        params = parse_qs(body)
        name = params.get("name", [""])[0]
        if not name:
            self._json({"ok": False, "error": "missing name"}, 400)
            return

        src = (self.server.source / name).resolve()
        if not str(src).startswith(str(self.server.source.resolve())) or not src.exists():
            self._json({"ok": False, "error": "file not found"}, 404)
            return

        try:
            if self.path == "/api/move":
                self.server.input_dir.mkdir(parents=True, exist_ok=True)
                dst = self.server.unique_target(self.server.input_dir, src.name)
                shutil.move(str(src), str(dst))
                self._json({"ok": True, "action": "moved", "target": str(dst)})
                return
            if self.path == "/api/discard":
                self.server.discard_dir.mkdir(parents=True, exist_ok=True)
                dst = self.server.unique_target(self.server.discard_dir, src.name)
                shutil.move(str(src), str(dst))
                self._json({"ok": True, "action": "discarded", "target": str(dst)})
                return
            self._json({"ok": False, "error": "unknown endpoint"}, 404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)

    def log_message(self, fmt: str, *args) -> None:
        return

    def _index_html(self) -> str:
        source = str(self.server.source)
        input_dir = str(self.server.input_dir)
        discard_dir = str(self.server.discard_dir)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Image Review Queue</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #111; color: #f3f3f3; }}
    .top {{ margin-bottom: 16px; }}
    .meta {{ color: #c8c8c8; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ background: #1b1b1b; border: 1px solid #2f2f2f; border-radius: 10px; overflow: hidden; }}
    img {{ width: 100%; height: 240px; object-fit: cover; background: #222; }}
    .pad {{ padding: 10px; }}
    .name {{ font-size: 12px; color: #dcdcdc; word-break: break-all; }}
    .dim {{ font-size: 12px; color: #9cc6ff; margin: 6px 0; }}
    .row {{ display: flex; gap: 8px; }}
    button {{ flex: 1; padding: 8px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }}
    .move {{ background: #2f9e44; color: #fff; }}
    .discard {{ background: #c92a2a; color: #fff; }}
    .counter {{ margin-top: 8px; color: #9de29d; }}
  </style>
</head>
<body>
  <div class="top">
    <h2>Scraped Image Review</h2>
    <div class="meta">Source: {source}</div>
    <div class="meta">Move target: {input_dir}</div>
    <div class="meta">Discard target: {discard_dir}</div>
    <div class="counter" id="counter"></div>
  </div>
  <div class="grid" id="grid"></div>
  <script>
    async function loadImages() {{
      const res = await fetch('/api/images');
      const data = await res.json();
      const images = data.images || [];
      document.getElementById('counter').textContent = `Remaining in queue: ${{images.length}}`;
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      for (const item of images) {{
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <img src="/image/${{encodeURIComponent(item.name)}}" loading="lazy"/>
          <div class="pad">
            <div class="name">${{item.name}}</div>
            <div class="dim">${{item.width}} x ${{item.height}} | ${{item.size_kb}} KB</div>
            <div class="row">
              <button class="move" data-name="${{item.name}}">Move to input</button>
              <button class="discard" data-name="${{item.name}}">Discard</button>
            </div>
          </div>`;
        grid.appendChild(card);
      }}
      bindButtons();
    }}

    async function postAction(path, name) {{
      const body = new URLSearchParams({{ name }});
      const res = await fetch(path, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
        body
      }});
      return res.json();
    }}

    function bindButtons() {{
      document.querySelectorAll('.move').forEach(btn => {{
        btn.onclick = async () => {{
          btn.disabled = true;
          await postAction('/api/move', btn.dataset.name);
          loadImages();
        }};
      }});
      document.querySelectorAll('.discard').forEach(btn => {{
        btn.onclick = async () => {{
          btn.disabled = true;
          await postAction('/api/discard', btn.dataset.name);
          loadImages();
        }};
      }});
    }}

    loadImages();
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review scraped images and move/discard them.")
    p.add_argument("--source", type=Path, required=True, help="Folder with scraped images to review.")
    p.add_argument("--input", type=Path, required=True, help="Target input folder for accepted images.")
    p.add_argument(
        "--discard",
        type=Path,
        default=None,
        help="Discard folder (default: <source>\\_discarded).",
    )
    p.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    input_dir = args.input.resolve()
    discard_dir = args.discard.resolve() if args.discard else (source / "_discarded")

    if not source.exists() or not source.is_dir():
        print(f"Source folder not found: {source}")
        return 1

    srv = ReviewServer((args.host, args.port), Handler, source, input_dir, discard_dir)
    print(f"Review UI: http://{args.host}:{args.port}")
    print(f"Source  : {source}")
    print(f"Input   : {input_dir}")
    print(f"Discard : {discard_dir}")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
