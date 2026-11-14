"""Video sources: files, webcams, RTSP/UDP streams and the synthetic scene."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from .synthetic import SyntheticScene


class ThreadedCapture:
    """Background frame grabber for live streams: always serves the newest frame
    so slow inference never builds up latency (critical for RTSP links from a UAV)."""

    def __init__(self, src: str | int):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame: np.ndarray | None = None
        self._ok = True
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._ok:
            ok, frame = self.cap.read()
            if not ok:
                self._ok = False
                self._new.set()
                break
            with self._lock:
                self._frame = frame
            self._new.set()

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            if not self._new.wait(timeout=5.0):
                break
            self._new.clear()
            with self._lock:
                frame = self._frame
            if frame is None:
                break
            yield frame
            if not self._ok:
                break

    def release(self) -> None:
        self._ok = False
        self.cap.release()


def _is_live(src: str) -> bool:
    return src.isdigit() or src.lower().startswith(("rtsp://", "rtmp://", "udp://", "http://", "https://"))


def open_source(src: str, max_frames: int | None = None) -> tuple[Iterator[np.ndarray], float]:
    """Return (frame iterator, nominal fps). `src` may be a file path, camera index,
    stream URL, or 'synthetic[:seed]'."""
    if src.startswith("synthetic"):
        seed = int(src.split(":", 1)[1]) if ":" in src else 7
        scene = SyntheticScene(seed=seed)
        n = max_frames or 600
        return (f for f, _ in scene.frames(n)), 30.0

    if _is_live(src):
        cap = ThreadedCapture(int(src) if src.isdigit() else src)
        return _limit(iter(cap), max_frames, cap.release), cap.fps

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {src!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen() -> Iterator[np.ndarray]:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame

    return _limit(gen(), max_frames, cap.release), fps


def _limit(it: Iterator[np.ndarray], n: int | None, on_close) -> Iterator[np.ndarray]:
    try:
        for i, frame in enumerate(it):
            if n is not None and i >= n:
                break
            yield frame
    finally:
        on_close()


class FpsMeter:
    def __init__(self, alpha: float = 0.1):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps
