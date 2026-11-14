"""Structured telemetry: one JSON object per frame (JSONL) plus lock events and a run summary.

JSONL is trivially consumed by pandas, Grafana/Loki or a ground-station bridge
(e.g. forwarding `lock.error` to a gimbal controller over MAVLink).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .lock import LockStatus
from .types import Track


class TelemetryWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")
        self.frames = 0
        self.events: list[dict[str, Any]] = []
        self.max_tracks = 0
        self.lock_frames = 0
        self.fps_sum = 0.0

    def write(self, idx: int, t: float, fps: float, tracks: list[Track], lock: LockStatus,
              latency_ms: float) -> None:
        self.frames += 1
        self.fps_sum += fps
        self.max_tracks = max(self.max_tracks, len(tracks))
        if lock.state.value == "LOCKED":
            self.lock_frames += 1
        for ev in lock.events:
            self.events.append(asdict(ev))
        if not self._fh:
            return
        rec = {
            "frame": idx,
            "t": round(t, 4),
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 2),
            "tracks": [
                {
                    "id": tr.track_id,
                    "label": tr.label,
                    "score": round(tr.score, 3),
                    "bbox": [round(v, 1) for v in tr.bbox],
                    "vel": [round(v, 2) for v in tr.velocity],
                }
                for tr in tracks
            ],
            "lock": {
                "state": lock.state.value,
                "track_id": lock.track_id,
                "candidate_id": lock.candidate_id,
                "progress": round(lock.progress, 3),
                "error": [round(e, 4) for e in lock.error] if lock.error else None,
            },
            "events": [asdict(e) for e in lock.events],
        }
        self._fh.write(json.dumps(rec) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "avg_fps": round(self.fps_sum / self.frames, 2) if self.frames else 0.0,
            "max_simultaneous_tracks": self.max_tracks,
            "locked_frames": self.lock_frames,
            "locks_acquired": sum(1 for e in self.events if e["kind"] == "lock_acquired"),
            "locks_lost": sum(1 for e in self.events if e["kind"] == "lock_lost"),
            "events": self.events,
        }

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
