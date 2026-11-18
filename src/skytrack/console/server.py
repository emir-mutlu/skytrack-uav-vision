"""Web operator console server - standard library only.

Endpoints
    GET  /                 single-page console (static/index.html, app.css, app.js)
    GET  /stream.mjpg      annotated camera feed (MJPEG)
    GET  /events           state stream (Server-Sent Events, 10 Hz)
    GET  /api/state        latest state snapshot (JSON)
    GET  /api/history      last two minutes of trend samples and vehicle track (JSON)
    GET  /api/snapshot.jpg latest annotated frame (JPEG)
    POST /api/command      operator command, e.g. {"cmd": "designate", "x": 0.5, "y": 0.4}
                           (refused with 403 when the session is read-only, e.g. a public demo)

Security: the server binds to 127.0.0.1 unless told otherwise. Commands must be
`application/json` POSTs, which a browser only sends cross-origin after a CORS preflight
this server never approves, and any Origin header must match the Host - so another web
page the operator has open cannot steer the gimbal.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .session import ConsoleSession

STATIC = Path(__file__).with_name("static")
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml"}
BOUNDARY = b"skytrackframe"
DISCONNECTED = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class _Handler(BaseHTTPRequestHandler):
    session: ConsoleSession
    server_version = "SkyTrackConsole/1.0"

    def log_message(self, fmt: str, *args) -> None:  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = urlsplit(self.path).path
        try:
            if path == "/api/state":
                return self._send(200, self.session.state_json().encode(), "application/json")
            if path == "/api/history":
                return self._json(200, self.session.history())
            if path == "/api/snapshot.jpg":
                jpeg = self.session.latest_jpeg()
                if jpeg is None:
                    return self._json(503, {"error": "No frame yet"})
                return self._send(200, jpeg, "image/jpeg")
            if path == "/stream.mjpg":
                return self._mjpeg()
            if path == "/events":
                return self._sse()
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            file = STATIC / name
            if "/" not in name and file.suffix in CONTENT_TYPES and file.is_file():
                return self._send(200, file.read_bytes(), CONTENT_TYPES[file.suffix])
            self._json(404, {"error": f"Not found: {path}"})
        except DISCONNECTED:
            pass

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/command":
            return self._json(404, {"ok": False, "error": "Not found"})
        if self.session.read_only:
            return self._json(403, {"ok": False, "error": "Read-only demo: run SkyTrack locally to control it"})
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            return self._json(415, {"ok": False, "error": "Send commands as application/json"})
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host", ""):
            return self._json(403, {"ok": False, "error": "Cross-origin command rejected"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 4096:
            return self._json(400, {"ok": False, "error": "Command body must be 1-4096 bytes"})
        try:
            cmd = json.loads(self.rfile.read(length))
        except ValueError:
            return self._json(400, {"ok": False, "error": "Command is not valid JSON"})
        if not isinstance(cmd, dict):
            return self._json(400, {"ok": False, "error": "Command must be a JSON object"})
        result = self.session.command(cmd)
        self._json(200 if result.get("ok") else 400, result)

    def _mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while self.session.running:
                jpeg, seq = self.session.wait_frame(last, 1.0)
                if jpeg is None or seq == last:
                    continue
                last = seq
                self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
        except DISCONNECTED:
            pass

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 1000\n\n")
            while self.session.running:
                self.wfile.write(b"data: " + self.session.state_json().encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.1)
        except DISCONNECTED:
            pass


class ConsoleServer:
    def __init__(self, session: ConsoleSession, host: str = "127.0.0.1", port: int = 8080):
        handler = type("ConsoleHandler", (_Handler,), {"session": session})
        self.session = session
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def start_background(self) -> ConsoleServer:
        self._thread = threading.Thread(target=self.serve_forever, name="console-http", daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
