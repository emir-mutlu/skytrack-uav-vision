"""Quantitative evaluation on the synthetic benchmark.

Reports detection precision / recall / F1 and CLEAR-MOT style tracking metrics
(MOTA, ID switches) plus per-frame latency, so detector/tracker changes can be
compared objectively instead of by eye.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .config import Config
from .detectors import build_detector
from .synthetic import SyntheticScene
from .tracking import MultiObjectTracker
from .types import BBox, iou_matrix


@dataclass
class EvalReport:
    frames: int
    gt_objects: int
    det_precision: float
    det_recall: float
    det_f1: float
    mota: float
    id_switches: int
    track_precision: float
    track_recall: float
    mean_latency_ms: float
    p95_latency_ms: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _match(gt: list[BBox], pred: list[BBox], thr: float) -> list[tuple[int, int]]:
    if not gt or not pred:
        return []
    m = iou_matrix(gt, pred)
    pairs, used_g, used_p = [], set(), set()
    for flat in np.argsort(-m, axis=None):
        g, p = np.unravel_index(flat, m.shape)
        if m[g, p] < thr:
            break
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        pairs.append((int(g), int(p)))
    return pairs


def evaluate(cfg: Config, frames: int = 600, seed: int = 7, iou_thr: float = 0.5) -> EvalReport:
    scene = SyntheticScene(seed=seed)
    detector = build_detector(cfg.detector)
    tracker = MultiObjectTracker(cfg.tracker)

    d_tp = d_fp = d_fn = 0
    t_tp = t_fp = t_fn = 0
    idsw = 0
    n_gt = 0
    last_id: dict[int, int] = {}
    lat: list[float] = []

    for frame, gt in scene.frames(frames):
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        tracks = tracker.update(dets)
        lat.append((time.perf_counter() - t0) * 1000)
        n_gt += len(gt)

        dm = _match(gt, [d.bbox for d in dets], iou_thr)
        d_tp += len(dm)
        d_fp += len(dets) - len(dm)
        d_fn += len(gt) - len(dm)

        tm = _match(gt, [t.bbox for t in tracks], iou_thr)
        t_tp += len(tm)
        t_fp += len(tracks) - len(tm)
        t_fn += len(gt) - len(tm)
        for g, p in tm:
            tid = tracks[p].track_id
            if g in last_id and last_id[g] != tid:
                idsw += 1
            last_id[g] = tid

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    dp, dr = ratio(d_tp, d_tp + d_fp), ratio(d_tp, d_tp + d_fn)
    return EvalReport(
        frames=frames,
        gt_objects=n_gt,
        det_precision=dp,
        det_recall=dr,
        det_f1=2 * dp * dr / (dp + dr) if dp + dr else 0.0,
        mota=1.0 - ratio(t_fn + t_fp + idsw, n_gt),
        id_switches=idsw,
        track_precision=ratio(t_tp, t_tp + t_fp),
        track_recall=ratio(t_tp, t_tp + t_fn),
        mean_latency_ms=float(np.mean(lat)) if lat else 0.0,
        p95_latency_ms=float(np.percentile(lat, 95)) if lat else 0.0,
    )
