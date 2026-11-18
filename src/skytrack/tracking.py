"""Multi-object tracker: constant-velocity Kalman filter + IoU association (SORT family).

Dependency-free (NumPy only) so it runs on any flight computer. Association uses the
Hungarian algorithm when SciPy is installed and falls back to greedy IoU matching.
"""

from __future__ import annotations

import numpy as np

from .config import TrackerConfig
from .types import BBox, Detection, Track, iou_matrix

try:  # optional, slightly better association under heavy clutter
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def _to_z(b: BBox) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


def _to_bbox(x: np.ndarray) -> BBox:
    cx, cy, w, h = x[:4]
    w, h = max(w, 1.0), max(h, 1.0)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanBox:
    """8-state constant-velocity Kalman filter over (cx, cy, w, h)."""

    def __init__(self, bbox: BBox):
        self.x = np.zeros(8)
        self.x[:4] = _to_z(bbox)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        w, h = self.x[2], self.x[3]
        s = max(w, h)
        self.P = np.diag([s, s, s, s, 10 * s, 10 * s, s, s]) ** 2 * 0.01
        self._q_scale = 0.05
        self._r_scale = 0.1

    def _Q(self) -> np.ndarray:
        s = max(self.x[2], self.x[3], 1.0) * self._q_scale
        return np.diag([s, s, s, s, s, s, s * 0.5, s * 0.5]) ** 2

    def _R(self) -> np.ndarray:
        s = max(self.x[2], self.x[3], 1.0) * self._r_scale
        return np.diag([s, s, s, s]) ** 2

    def predict(self) -> BBox:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()
        return _to_bbox(self.x)

    def update(self, bbox: BBox) -> None:
        z = _to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self._R()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def bbox(self) -> BBox:
        return _to_bbox(self.x)


def associate(
    tracks: list[BBox], dets: list[BBox], thr: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matches, unmatched_track_idx, unmatched_det_idx)."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    ious = iou_matrix(tracks, dets)
    pairs: list[tuple[int, int]] = []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-ious)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if ious[r, c] >= thr]
    else:
        used_t, used_d = set(), set()
        for flat in np.argsort(-ious, axis=None):
            t, d = np.unravel_index(flat, ious.shape)
            if ious[t, d] < thr:
                break
            if t in used_t or d in used_d:
                continue
            used_t.add(int(t))
            used_d.add(int(d))
            pairs.append((int(t), int(d)))
    mt = {p[0] for p in pairs}
    md = {p[1] for p in pairs}
    return (
        pairs,
        [i for i in range(len(tracks)) if i not in mt],
        [j for j in range(len(dets)) if j not in md],
    )


class _Tracklet:
    def __init__(self, tid: int, det: Detection):
        self.kf = KalmanBox(det.bbox)
        self.track = Track(tid, det.bbox, det.score, det.cls, det.label)


class MultiObjectTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self._tracklets: list[_Tracklet] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracklets.clear()
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        cfg = self.cfg
        predicted = [t.kf.predict() for t in self._tracklets]
        for t in self._tracklets:
            t.track.age += 1
            t.track.time_since_update += 1

        matches, unmatched_t, unmatched_d = associate(predicted, [d.bbox for d in detections], cfg.iou_threshold)
        for ti in unmatched_t:
            # Coasting: hold the box size. A missed detection says nothing about size, and a
            # target shrinking under partial occlusion would otherwise collapse to a point and
            # no longer overlap the detection when it reappears.
            self._tracklets[ti].kf.x[6:8] = 0.0

        for ti, di in matches:
            tl, det = self._tracklets[ti], detections[di]
            tl.kf.update(det.bbox)
            tr = tl.track
            tr.score = 0.7 * tr.score + 0.3 * det.score
            tr.cls, tr.label = det.cls, det.label
            tr.hits += 1
            tr.time_since_update = 0
            if tr.hits >= cfg.min_hits:
                tr.confirmed = True

        for di in unmatched_d:
            tl = _Tracklet(self._next_id, detections[di])
            self._next_id += 1
            if cfg.min_hits <= 1:
                tl.track.confirmed = True
            self._tracklets.append(tl)

        alive: list[_Tracklet] = []
        for tl in self._tracklets:
            tr = tl.track
            if tr.time_since_update > cfg.max_age:
                continue
            if not tr.confirmed and tr.time_since_update > 1:  # drop flicker tentatives fast
                continue
            tr.bbox = tl.kf.bbox
            tr.velocity = (float(tl.kf.x[4]), float(tl.kf.x[5]))
            cx, cy = tr.center
            tr.history.append((int(cx), int(cy)))
            if len(tr.history) > cfg.history:
                del tr.history[: len(tr.history) - cfg.history]
            alive.append(tl)
        self._tracklets = alive
        return [tl.track for tl in alive if tl.track.confirmed]

    @property
    def all_tracks(self) -> list[Track]:
        return [tl.track for tl in self._tracklets]
