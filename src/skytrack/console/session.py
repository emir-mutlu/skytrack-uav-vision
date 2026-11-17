"""Console session: runs the perception pipeline in a background thread and publishes the
latest annotated frame (JPEG) and a JSON state snapshot for the web console.

Operator commands arrive from HTTP threads and are queued; they are applied on the
pipeline thread between frames, so the tracker and lock state are never touched by two
threads at once.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..geo import geolocate
from ..hud import draw_hud
from ..pipeline import FrameResult, Pipeline

CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
LOCK_COMMANDS = {"designate", "designate_id", "release", "auto_designation"}


class ConsoleSession:
    def __init__(self, cfg: Config, frames: Iterator[np.ndarray], src_fps: float = 30.0,
                 live: bool = False, source_label: str = "", jpeg_quality: int = 82,
                 pipeline: Pipeline | None = None):
        self.cfg = cfg
        self.pipe = pipeline or Pipeline(cfg)
        self.frames = iter(frames)
        self.src_fps = src_fps or 30.0
        self.live = live
        self.source_label = source_label
        self.jpeg_quality = jpeg_quality
        self.running = False
        self.paused = False
        self.ended = False
        self.seq = 0
        self.events: deque[dict[str, Any]] = deque(maxlen=40)
        # (monotonic time, |boresight error| or None, detector latency ms, lat, lon) per frame
        self._history: deque[tuple] = deque(maxlen=3600)
        self._cmds: queue.Queue[dict[str, Any]] = queue.Queue()
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._state_json = "{}"
        self._thread: threading.Thread | None = None
        self._last: FrameResult | None = None
        self._shape: tuple[int, ...] = (720, 1280, 3)
        self._link_up: bool | None = None
        self._t = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> ConsoleSession:
        self.running = True
        if self.pipe.link is not None:
            self.pipe.link.start()
        self._thread = threading.Thread(target=self._loop, name="console-pipeline", daemon=True)
        self._thread.start()
        self._event("session_start", detail=self.source_label)
        return self

    def stop(self) -> None:
        self.running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        link = self.pipe.link
        if link is not None:
            if self.pipe.gimbal_ctl is not None and self.pipe.gimbal_ctl.active:
                link.send_gimbal_rates(0.0, 0.0)
            link.stop()

    # ------------------------------------------------------------------ thread-safe API
    def wait_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._cond:
            if self.seq == last_seq and self.running:
                self._cond.wait(timeout)
            return self._jpeg, self.seq

    def latest_jpeg(self) -> bytes | None:
        with self._cond:
            return self._jpeg

    def state_json(self) -> str:
        with self._cond:
            return self._state_json

    def history(self, seconds: float = 120.0) -> dict[str, Any]:
        """Recent samples so a freshly opened console starts with full trend charts."""
        now = time.monotonic()
        with self._cond:
            rows = [r for r in self._history if now - r[0] <= seconds]
        return {
            "err": [[round(now - t, 2), None if e is None else round(e, 4)] for t, e, *_ in rows[::3]],
            "lat": [[round(now - t, 2), round(ms, 2)] for t, _, ms, *_ in rows[::3]],
            "trail": [[round(now - t, 1), la, lo] for t, _, _, la, lo in rows[::15] if la is not None],
        }

    def command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Validate an operator command and queue it for the pipeline thread."""
        kind = cmd.get("cmd")
        if kind == "designate":
            x, y = cmd.get("x"), cmd.get("y")
            if not all(isinstance(v, int | float) and 0.0 <= v <= 1.0 for v in (x, y)):
                return {"ok": False, "error": "designate needs x and y between 0 and 1"}
        elif kind == "designate_id":
            if not isinstance(cmd.get("id"), int):
                return {"ok": False, "error": "designate_id needs an integer track id"}
        elif kind in ("follow", "auto_designation"):
            if not isinstance(cmd.get("on"), bool):
                return {"ok": False, "error": f"{kind} needs on: true or false"}
            if kind == "follow" and self.pipe.link is None:
                return {"ok": False, "error": "Gimbal follow needs a flight-controller link: start with --mavlink "
                                              "or the sim source"}
        elif kind not in ("release", "pause", "resume", "toggle_pause"):
            return {"ok": False, "error": f"Unknown command: {kind!r}"}
        if kind in LOCK_COMMANDS and not self.cfg.lock.enabled:
            return {"ok": False, "error": "Target lock is disabled in this mission profile"}
        self._cmds.put(dict(cmd))
        return {"ok": True}

    # ------------------------------------------------------------------ pipeline thread
    def _event(self, kind: str, track_id: int | None = None, detail: str = "") -> None:
        self.events.append({"utc": time.strftime("%H:%M:%S", time.gmtime()), "kind": kind,
                            "track_id": track_id, "detail": detail})

    def _apply(self, cmd: dict[str, Any]) -> None:
        kind, pipe, t = cmd["cmd"], self.pipe, self._t
        locker = pipe.locker
        if kind == "designate":
            if not self._hit(cmd["x"], cmd["y"]):
                self._event("no_track", detail="No track under the cursor")
                return
            h, w = self._shape[:2]
            pipe.designate_at(cmd["x"] * w, cmd["y"] * h, t)
            self._event("designated", locker.designated_id)
        elif kind == "designate_id":
            if any(tr.track_id == cmd["id"] for tr in pipe._last_tracks):
                locker.designate(cmd["id"], t)
                self._event("designated", cmd["id"])
            else:
                self._event("no_track", cmd["id"], "Track no longer active")
        elif kind == "release":
            tid = locker.target_id or locker.designated_id
            locker.release(t)
            self._event("released", tid)
        elif kind in ("pause", "resume", "toggle_pause"):
            self.paused = {"pause": True, "resume": False}.get(kind, not self.paused)
            if self.paused:
                self._stop_gimbal()
            self._event("paused" if self.paused else "resumed")
        elif kind == "follow":
            self.cfg.mavlink.gimbal.follow = cmd["on"]
            if not cmd["on"]:
                self._stop_gimbal()
            self._event("follow_on" if cmd["on"] else "follow_off")
        elif kind == "auto_designation":
            self.cfg.lock.require_operator_confirm = not cmd["on"]
            self._event("auto_on" if cmd["on"] else "auto_off")

    def _hit(self, x: float, y: float) -> bool:
        h, w = self._shape[:2]
        return any(tr.bbox[0] <= x * w <= tr.bbox[2] and tr.bbox[1] <= y * h <= tr.bbox[3]
                   for tr in self.pipe._last_tracks)

    def _stop_gimbal(self) -> None:
        ctl, link = self.pipe.gimbal_ctl, self.pipe.link
        if ctl is not None and link is not None and ctl.active:
            link.send_gimbal_rates(0.0, 0.0)
            ctl.active = False
            ctl.reset()

    def _drain(self) -> bool:
        applied = False
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return applied
            self._apply(cmd)
            applied = True

    def _publish(self, jpeg: bytes | None) -> None:
        state = self._build_state(self._last) if self._last is not None else {}
        text = json.dumps(state, separators=(",", ":"))
        with self._cond:
            if jpeg is not None:
                self._jpeg = jpeg
            self.seq += 1
            self._state_json = text
            self._cond.notify_all()

    def _loop(self) -> None:
        period = 1.0 / self.src_fps
        t_start = due = time.perf_counter()
        idx = 0
        while self.running:
            if self._drain() and (self.paused or self.ended):
                self._publish(None)
            if self.paused or self.ended:
                time.sleep(0.05)
                continue
            frame = next(self.frames, None)
            if frame is None:
                self.ended = True
                self._event("source_end", detail="Video source finished")
                self._publish(None)
                continue
            self._t = (time.perf_counter() - t_start) if self.live else idx / self.src_fps
            res = self.pipe.process(frame, idx, self._t)
            self._last, self._shape = res, frame.shape
            for ev in res.lock.events:
                self._event(ev.kind, ev.track_id, f"{ev.duration:.1f} s" if ev.duration else "")
            vs = res.extras.get("vehicle")
            if vs is not None:
                up = vs.connected(self.cfg.mavlink.heartbeat_timeout)
                if up != self._link_up and (self._link_up is not None or up):
                    self._event("link_up" if up else "link_lost", detail=vs.mode if up else "")
                self._link_up = up
            err = res.lock.error
            la = lo = None
            if vs is not None and vs.has_position and self._link_up:
                la, lo = vs.lat, vs.lon
            with self._cond:
                self._history.append((time.monotonic(), None if err is None else math.hypot(*err),
                                      res.latency_ms, la, lo))
            vis = draw_hud(frame.copy(), res.tracks, res.lock, self.pipe.locker, res.fps, res.latency_ms,
                           self.pipe.detector.name, draw_trails=self.cfg.output.draw_trails, panels=False)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            self._publish(buf.tobytes() if ok else None)
            idx += 1
            if not self.live:
                due += period
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    due = time.perf_counter()  # running behind: do not try to catch up

    # ------------------------------------------------------------------ state snapshot
    def _footprint(self, vs, w: int, h: int) -> list[list[float] | None]:
        cam = self.cfg.camera
        gp = vs.gimbal_pitch_deg if vs.gimbal_pitch_deg is not None else cam.default_pitch_deg
        gy = vs.gimbal_yaw_deg if vs.gimbal_yaw_deg is not None else cam.default_yaw_deg
        out: list[list[float] | None] = []
        for err in CORNERS:
            g = geolocate(err, vs.lat, vs.lon, vs.alt_rel, (vs.roll, vs.pitch, vs.yaw), gp, gy, cam.hfov_deg,
                          h / w, cam.stabilized, terrain=self.pipe.terrain, cam_alt_amsl=vs.alt_amsl)
            out.append(None if g is None else [round(g.north_m, 1), round(g.east_m, 1)])
        return out

    def _build_state(self, res: FrameResult) -> dict[str, Any]:
        h, w = self._shape[:2]
        pipe, lk, cfg = self.pipe, res.lock, self.cfg
        geo = res.extras.get("track_geo", {})
        tracks = []
        for tr in res.tracks:
            role = ("target" if tr.track_id == lk.track_id
                    else "candidate" if tr.track_id == lk.candidate_id else "track")
            x1, y1, x2, y2 = tr.bbox
            g = geo.get(tr.track_id)
            tracks.append({
                "id": tr.track_id, "label": tr.label, "score": round(tr.score, 3), "role": role,
                "bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                "age_s": round(tr.age / self.src_fps, 1),
                "speed_px_s": round(math.hypot(*tr.velocity) * self.src_fps, 1),
                "geo": None if g is None else {"lat": round(g.lat, 7), "lon": round(g.lon, 7),
                                               "range_m": round(g.ground_range_m, 1), "method": g.method},
            })
        order = {"target": 0, "candidate": 1, "track": 2}
        tracks.sort(key=lambda d: (order[d["role"]], d["id"]))

        vehicle, footprint = None, None
        vs = res.extras.get("vehicle")
        if vs is not None:
            vehicle = vs.as_dict()
            vehicle["connected"] = vs.connected(cfg.mavlink.heartbeat_timeout)
            if vehicle["connected"] and vs.has_position:
                footprint = self._footprint(vs, w, h)
        tg = res.extras.get("target_geo")
        cmd = res.extras.get("gimbal_cmd")
        return {
            "seq": self.seq + 1, "t": round(res.t, 2), "frame": res.index, "fps": round(res.fps, 1),
            "latency_ms": round(res.latency_ms, 2), "paused": self.paused, "ended": self.ended,
            "source": self.source_label, "profile": cfg.profile, "detector": pipe.detector.name, "size": [w, h],
            "lock": {
                "enabled": cfg.lock.enabled, "state": lk.state.value, "track_id": lk.track_id,
                "candidate_id": lk.candidate_id, "designated_id": pipe.locker.designated_id,
                "progress": round(lk.progress, 3), "duration": round(lk.lock_duration, 2),
                "error": None if lk.error is None else [round(e, 4) for e in lk.error],
                "auto": not cfg.lock.require_operator_confirm, "acquire_s": cfg.lock.acquire_seconds,
            },
            "tracks": tracks,
            "vehicle": vehicle,
            "target_geo": None if tg is None else {"lat": round(tg.lat, 7), "lon": round(tg.lon, 7),
                                                   "range_m": round(tg.ground_range_m, 1), "method": tg.method},
            "gimbal": {"available": pipe.link is not None, "follow": cfg.mavlink.gimbal.follow,
                       "active": bool(res.extras.get("gimbal_follow")),
                       "cmd": None if cmd is None else [round(v, 2) for v in cmd]},
            "footprint": footprint,
            "events": list(self.events)[-30:],
        }
