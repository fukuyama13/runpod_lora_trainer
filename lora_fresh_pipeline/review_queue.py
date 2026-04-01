from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import shutil
from urllib.parse import parse_qs, urlparse

from PIL import Image


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


class App(ThreadingHTTPServer):
    def __init__(self, addr, handler, source: Path, input_dir: Path, discard_dir: Path):
        super().__init__(addr, handler)
        self.source = source
        self.input_dir = input_dir
        self.discard_dir = discard_dir


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, data: dict, code: int = 200) -> None:
        blob = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:
        p = urlparse(self.path)
        if p.path == "/":
            return self._index()
        if p.path == "/api/images":
            items = []
            for f in sorted(self.server.source.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED:
                    try:
                        with Image.open(f) as im:
                            w, h = im.size
                    except Exception:
                        w, h = 0, 0
                    items.append({"name": f.name, "w": w, "h": h})
            return self._json({"images": items})
        if p.path.startswith("/img/"):
            name = p.path[len("/img/") :]
            fp = (self.server.source / name).resolve()
            if not str(fp).startswith(str(self.server.source.resolve())) or not fp.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n).decode("utf-8")
        name = parse_qs(body).get("name", [""])[0]
        src = (self.server.source / name).resolve()
        if not str(src).startswith(str(self.server.source.resolve())) or not src.exists():
            return self._json({"ok": False}, 404)
        if self.path == "/api/move":
            dst_dir = self.server.input_dir
        elif self.path == "/api/discard":
            dst_dir = self.server.discard_dir
        else:
            return self._json({"ok": False}, 404)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        i = 2
        while dst.exists():
            dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
            i += 1
        shutil.move(str(src), str(dst))
        self._json({"ok": True})

    def _index(self) -> None:
        html = """<!doctype html><html><head><meta charset='utf-8'><title>Review Queue</title>
<style>body{font-family:Arial;background:#111;color:#eee;margin:18px}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}.c{background:#1b1b1b;border:1px solid #333;border-radius:10px;overflow:hidden}img{width:100%;height:220px;object-fit:cover}.p{padding:8px}.r{display:flex;gap:8px}button{flex:1;padding:7px;border:0;border-radius:7px;color:#fff;cursor:pointer}.m{background:#2f9e44}.d{background:#c92a2a}</style>
</head><body><h3>Scraped Image Review</h3><div id='count'></div><div class='g' id='g'></div>
<script>
async function load(){const r=await fetch('/api/images');const j=await r.json();document.getElementById('count').textContent='Remaining: '+j.images.length;const g=document.getElementById('g');g.innerHTML='';
for (const it of j.images){const d=document.createElement('div');d.className='c';d.innerHTML=`<img src="/img/${encodeURIComponent(it.name)}"><div class='p'><div>${it.name}</div><div>${it.w}x${it.h}</div><div class='r'><button class='m' data-n='${it.name}'>Move</button><button class='d' data-n='${it.name}'>Discard</button></div></div>`;g.appendChild(d);}
document.querySelectorAll('.m').forEach(b=>b.onclick=()=>act('/api/move',b.dataset.n));
document.querySelectorAll('.d').forEach(b=>b.onclick=()=>act('/api/discard',b.dataset.n));}
async function act(path,name){await fetch(path,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({name})});load();}
load();
</script></body></html>"""
        blob = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--discard", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    source = args.source.resolve()
    discard = args.discard.resolve() if args.discard else (source / "_discarded")
    srv = App((args.host, args.port), H, source, args.input.resolve(), discard)
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Review UI: http://{display_host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

