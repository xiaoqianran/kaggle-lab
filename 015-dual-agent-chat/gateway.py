#!/usr/bin/env python3
"""Local / VPS gateway: static web UI + CORS-friendly Model Proxy relay.

  python gateway.py
  # http://127.0.0.1:8765/

Reads MODEL_PROXY_* from env or ../../.env.model-proxy (kaggle-lab root).
Browser talks to /api/openai/* ; gateway forwards to {MODEL_PROXY_URL}/openapi/*
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
WEB_DIST = ROOT / "web" / "dist"
ENV_FILE = LAB / ".env.model-proxy"
PORT = int(os.environ.get("PORT", "8765"))
HOST = os.environ.get("HOST", "0.0.0.0")


def load_env() -> None:
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    token = Path.home() / ".kaggle" / "access_token"
    if token.is_file() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = token.read_text(encoding="utf-8").strip()


def proxy_target() -> tuple[str, str]:
    url = os.environ.get("MODEL_PROXY_URL", "").rstrip("/")
    key = os.environ.get("MODEL_PROXY_API_KEY", "")
    if not url or not key:
        raise RuntimeError(
            f"缺少 MODEL_PROXY_URL / MODEL_PROXY_API_KEY。先运行:\n"
            f"  cd {LAB} && kaggle b auth -y --env-file .env.model-proxy"
        )
    return f"{url}/openapi", key


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIST), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/api/health"):
            self._json(200, {"ok": True, "service": "015-gateway"})
            return
        # SPA fallback
        if not self.path.startswith("/api/") and not Path(WEB_DIST, self.path.lstrip("/")).is_file():
            if not self.path.startswith("/assets/"):
                self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/api/openai/"):
            return self._proxy()
        self.send_error(404)

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        # Allow browser to override key via Authorization; else use server env
        auth = self.headers.get("Authorization") or ""
        try:
            base, env_key = proxy_target()
        except RuntimeError as e:
            self._json(503, {"error": str(e)})
            return
        if auth.lower().startswith("bearer ") and len(auth) > 20:
            maybe = auth.split(" ", 1)[1].strip()
            key = env_key if (not maybe or maybe == "use-server") else maybe
        else:
            key = env_key
        sub = self.path[len("/api/openai") :]  # e.g. /chat/completions
        url = f"{base}{sub}"
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._json(502, {"error": str(e)})

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    load_env()
    if not WEB_DIST.is_dir():
        print(f"缺少前端构建产物: {WEB_DIST}")
        print("请先: cd web && npm i && npm run build")
        return 1
    # For local gateway, SPA should call /api/openai (same origin).
    # Built pages use base /kaggle-lab/ — still fine if assets relative.
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"015 dual-agent gateway  http://127.0.0.1:{PORT}/")
    print(f"  static: {WEB_DIST}")
    print(f"  proxy:  /api/openai/* → MODEL_PROXY openapi")
    try:
        proxy_target()
        print("  Model Proxy: configured")
    except RuntimeError as e:
        print(f"  Model Proxy: NOT ready ({e.splitlines()[0]})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
