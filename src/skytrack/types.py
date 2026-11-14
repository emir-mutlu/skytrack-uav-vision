"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)


@dataclass
class Detection:
    bbox: BBox
    score: float
    cls: int = 0
    label: str = "target"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    score: float
    cls: int
    label: str
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)  # px/frame
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return x2 - x1, y2 - y1


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def iou_matrix(a: list[BBox], b: list[BBox]) -> np.ndarray:
    """Vectorised pairwise IoU, shape (len(a), len(b))."""
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A = np.asarray(a, dtype=np.float32)[:, None, :]
    B = np.asarray(b, dtype=np.float32)[None, :, :]
    ix1 = np.maximum(A[..., 0], B[..., 0])
    iy1 = np.maximum(A[..., 1], B[..., 1])
    ix2 = np.minimum(A[..., 2], B[..., 2])
    iy2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    area_b = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)
