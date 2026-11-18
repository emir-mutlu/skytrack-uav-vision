"""Web operator console: HTTP endpoints, live streams, commands and request hardening."""

import http.client
import json
import time

import pytest

from skytrack.config import load_config
from skytrack.console import ConsoleServer, ConsoleSession
from skytrack.synthetic import SyntheticScene


@pytest.fixture(scope="module")
def console():
    cfg = load_config("configs/demo.yaml", {"output": {"show": False}})
    scene = SyntheticScene(seed=7)
    session = ConsoleSession(cfg, (f for f, _ in scene.frames(100_000)), 30.0, False, "Synthetic").start()
    server = ConsoleServer(session, "127.0.0.1", 0).start_background()
    port = server.httpd.server_address[1]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not json.loads(session.state_json()).get("tracks"):
        time.sleep(0.1)
    yield port, session
    server.shutdown()
    session.stop()


def _get(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    return resp.status, resp.getheader("Content-Type", ""), resp.read(), conn


def _post(port, body, content_type="application/json", origin=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": content_type}
    if origin:
        headers["Origin"] = origin
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    conn.request("POST", "/api/command", body=raw, headers=headers)
    resp = conn.getresponse()
    data = json.loads(resp.read() or b"{}")
    conn.close()
    return resp.status, data


def _state(port):
    return json.loads(_get(port, "/api/state")[2])


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def test_static_ui_is_served(console):
    port, _ = console
    status, ctype, body, _ = _get(port, "/")
    assert status == 200 and ctype.startswith("text/html") and b"SkyTrack GCS" in body
    for path, kind in (("/app.js", "javascript"), ("/app.css", "text/css")):
        status, ctype, body, _ = _get(port, path)
        assert status == 200 and kind in ctype and len(body) > 1000
    assert _get(port, "/../pyproject.toml")[0] == 404
    assert _get(port, "/console/server.py")[0] == 404


def test_state_snapshot_and_history(console):
    port, _ = console
    s = _state(port)
    assert {"lock", "tracks", "size", "fps", "events", "gimbal"} <= s.keys()
    assert s["size"] == [960, 540] and s["tracks"]
    t = s["tracks"][0]
    assert {"id", "label", "score", "role", "bbox", "age_s"} <= t.keys()
    assert all(0.0 <= v <= 1.0 for v in t["bbox"])
    status, _, jpeg, _ = _get(port, "/api/snapshot.jpg")
    assert status == 200 and jpeg[:2] == b"\xff\xd8"
    hist = json.loads(_get(port, "/api/history")[2])
    assert hist["lat"] and len(hist["lat"][0]) == 2


def test_mjpeg_and_sse_streams(console):
    port, _ = console
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/stream.mjpg")
    resp = conn.getresponse()
    assert resp.status == 200 and "multipart/x-mixed-replace" in resp.getheader("Content-Type")
    chunk = resp.read(4096)
    assert b"--skytrackframe" in chunk and b"\xff\xd8" in chunk
    conn.close()

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert resp.getheader("Content-Type") == "text/event-stream"
    line = b""
    while not line.startswith(b"data: "):
        line = resp.fp.readline()
    assert "lock" in json.loads(line[6:])
    conn.close()


def test_designate_pause_and_release(console):
    port, _ = console
    track = _state(port)["tracks"][0]
    x1, y1, x2, y2 = track["bbox"]
    assert _post(port, {"cmd": "designate", "x": (x1 + x2) / 2, "y": (y1 + y2) / 2}) == (200, {"ok": True})
    assert _wait(lambda: _state(port)["lock"]["designated_id"] == track["id"])
    assert any(e["kind"] == "designated" for e in _state(port)["events"])

    assert _post(port, {"cmd": "pause"})[0] == 200
    assert _wait(lambda: _state(port)["paused"])
    frame = _state(port)["frame"]
    time.sleep(0.3)
    assert _state(port)["frame"] == frame  # frozen while paused
    assert _post(port, {"cmd": "resume"})[0] == 200
    assert _wait(lambda: _state(port)["frame"] > frame)

    assert _post(port, {"cmd": "release"})[0] == 200
    assert _wait(lambda: _state(port)["lock"]["designated_id"] is None)


def test_auto_designation_toggle(console):
    port, session = console
    assert _post(port, {"cmd": "auto_designation", "on": False})[0] == 200
    assert _wait(lambda: _state(port)["lock"]["auto"] is False)
    assert session.cfg.lock.require_operator_confirm is True
    assert _post(port, {"cmd": "auto_designation", "on": True})[0] == 200
    assert _wait(lambda: _state(port)["lock"]["auto"] is True)


def test_command_validation_and_hardening(console):
    port, _ = console
    assert _post(port, {"cmd": "designate", "x": 3, "y": 0.5})[0] == 400
    assert _post(port, {"cmd": "self_destruct"})[0] == 400
    status, data = _post(port, {"cmd": "follow", "on": True})
    assert status == 400 and "flight-controller link" in data["error"]
    assert _post(port, {"cmd": "release"}, content_type="text/plain")[0] == 415  # no CSRF "simple request"
    assert _post(port, {"cmd": "release"}, origin="http://evil.example")[0] == 403
    assert _post(port, b"not json")[0] == 400
    assert _post(port, b"[1, 2]")[0] == 400
