"""End-to-end pipeline: source -> detector -> tracker -> target lock -> HUD / telemetry / video."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import Config
from .detectors import Detector, build_detector
from .geo import GeoPoint, geolocate
from .gimbal import GimbalController
from .hud import draw_hud
from .lock import LockStatus, TargetLock
from .sources import FpsMeter
from .telemetry import TelemetryWriter
from .tracking import MultiObjectTracker
from .types import Detection, Track

WINDOW = "SkyTrack"


@dataclass
class FrameResult:
    index: int
    t: float
    detections: list[Detection]
    tracks: list[Track]
    lock: LockStatus
    latency_ms: float
    fps: float
    extras: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: Config, detector: Detector | None = None):
        self.cfg = cfg
        self.detector = detector or build_detector(cfg.detector)
        self.tracker = MultiObjectTracker(cfg.tracker)
        self.locker = TargetLock(cfg.lock)
        self.fps_meter = FpsMeter()
        self._last_tracks: list[Track] = []
        self.link = None
        self.gimbal_ctl: GimbalController | None = None
        if cfg.mavlink.enabled:
            from .vehicle import MavlinkLink  # optional dependency (pymavlink)

            self.link = MavlinkLink(cfg.mavlink)

    # ------------------------------------------------------------------ core
    def process(self, frame: np.ndarray, index: int, t: float) -> FrameResult:
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        tracks = self.tracker.update(dets)
        lock = self.locker.update(tracks, frame.shape, t)
        latency = (time.perf_counter() - t0) * 1000.0
        self._last_tracks = tracks
        res = FrameResult(index, t, dets, tracks, lock, latency, self.fps_meter.tick())
        if self.link is not None:
            self._vehicle_step(frame.shape, res)
        return res

    # ------------------------------------------------------------------ vehicle integration
    def _vehicle_step(self, shape: tuple[int, ...], res: FrameResult) -> None:
        """Geolocate tracks with the live vehicle state and drive the gimbal follow loop."""
        cfg, cam = self.cfg, self.cfg.camera
        vs = self.link.state
        connected = vs.connected(cfg.mavlink.heartbeat_timeout)
        res.extras["vehicle"] = vs
        h, w = shape[:2]
        aspect = h / w

        if cfg.mavlink.geolocate and connected and vs.has_position:
            gp = vs.gimbal_pitch_deg if vs.gimbal_pitch_deg is not None else cam.default_pitch_deg
            gy = vs.gimbal_yaw_deg if vs.gimbal_yaw_deg is not None else cam.default_yaw_deg
            geo: dict[int, GeoPoint] = {}
            for tr in res.tracks:
                pt = geolocate(TargetLock.centre_error(tr, shape), vs.lat, vs.lon, vs.alt_rel,
                               (vs.roll, vs.pitch, vs.yaw), gp, gy, cam.hfov_deg, aspect, cam.stabilized)
                if pt is not None:
                    geo[tr.track_id] = pt
            res.extras["track_geo"] = geo
            if res.lock.track_id in geo:
                res.extras["target_geo"] = geo[res.lock.track_id]

        if cfg.mavlink.gimbal.follow:
            if self.gimbal_ctl is None:
                self.gimbal_ctl = GimbalController(cfg.mavlink.gimbal, cam.hfov_deg, aspect)
            cmd = self.gimbal_ctl.update(res.lock.error if connected else None, res.t)
            if cmd is not None:
                self.link.send_gimbal_rates(*cmd)
                res.extras["gimbal_cmd"] = cmd
            res.extras["gimbal_follow"] = self.gimbal_ctl.active

    # ------------------------------------------------------------------ operator input
    def designate_at(self, x: int, y: int, t: float) -> None:
        hits = [tr for tr in self._last_tracks
                if tr.bbox[0] <= x <= tr.bbox[2] and tr.bbox[1] <= y <= tr.bbox[3]]
        if hits:
            self.locker.designate(min(hits, key=lambda tr: tr.size[0] * tr.size[1]).track_id, t)

    def cycle_designation(self, t: float) -> None:
        ids = sorted(tr.track_id for tr in self._last_tracks)
        if not ids:
            return
        cur = self.locker.designated_id
        nxt = next((i for i in ids if cur is None or i > cur), ids[0])
        self.locker.designate(nxt, t)

    # ------------------------------------------------------------------ run loop
    def run(self, frames: Iterable[np.ndarray], src_fps: float = 30.0, live: bool = False,
            show: bool | None = None) -> dict[str, Any]:
        out = self.cfg.output
        show = out.show if show is None else show
        telemetry = TelemetryWriter(out.telemetry)
        writer: cv2.VideoWriter | None = None
        state = {"t": 0.0, "click": None}
        start = time.perf_counter()
        paused = False
        if self.link is not None:
            self.link.start()

        if show:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(
                WINDOW, lambda ev, x, y, *_: state.update(click=(x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None
            )

        try:
            it = iter(frames)
            idx = 0
            frame = None
            res = None
            while True:
                if not paused or frame is None:
                    frame = next(it, None)
                    if frame is None:
                        break
                    t = (time.perf_counter() - start) if live else idx / src_fps
                    state["t"] = t
                    res = self.process(frame, idx, t)
                    telemetry.write(idx, t, res.fps, res.tracks, res.lock, res.latency_ms, res.extras)
                    idx += 1

                assert res is not None
                vis = draw_hud(frame.copy(), res.tracks, res.lock, self.locker, res.fps,
                               res.latency_ms, self.detector.name, out.draw_trails, paused,
                               profile=self.cfg.profile, frame_idx=res.index,
                               vehicle=res.extras.get("vehicle"), target_geo=res.extras.get("target_geo"),
                               gimbal_follow=bool(res.extras.get("gimbal_follow")),
                               link_timeout=self.cfg.mavlink.heartbeat_timeout)

                if out.save_video and not paused:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(out.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 src_fps, (w, h))
                    writer.write(vis)

                if show:
                    cv2.imshow(WINDOW, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif key in (9, ord("t")):  # TAB
                        self.cycle_designation(state["t"])
                    elif key == ord("r"):
                        self.locker.release(state["t"])
                    if state["click"] is not None:
                        self.designate_at(*state["click"], state["t"])
                        state["click"] = None
        finally:
            if self.link is not None:
                if self.gimbal_ctl is not None and self.gimbal_ctl.active:
                    self.link.send_gimbal_rates(0.0, 0.0)  # leave the gimbal still
                self.link.stop()
            telemetry.close()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        return telemetry.summary()
