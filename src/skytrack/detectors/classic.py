"""Classic (non-learning) square-target detector.

A modernised version of the 2015 PyImageSearch "Target acquired" approach:
contours -> polygon approximation -> geometric filters. Improvements over the original:

* CLAHE contrast normalisation (robust to haze / sun glare in aerial footage)
* automatic Canny thresholds from the image median instead of fixed values
* morphological closing so broken edges still produce closed contours
* a continuous confidence score instead of a binary yes/no
* containment-aware NMS so nested squares produce a single detection

It needs no GPU and no training data, which makes it useful as a fallback
on low-power flight computers and as a reference baseline for the deep model.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import DetectorConfig
from ..types import Detection
from .base import Detector


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper - lower < 30:  # flat images: fall back to a sane band
        lower, upper = 30, 120
    return cv2.Canny(gray, lower, upper)


def nms_contained(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """Greedy NMS that also suppresses boxes mostly contained in a kept box.

    Larger boxes are visited first, so for nested-square markers the outermost
    square (the true target extent) is kept and inner rings are suppressed."""
    dets = sorted(
        dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) * d.score, reverse=True
    )
    kept: list[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        area = (x2 - x1) * (y2 - y1)
        suppressed = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k.bbox
            iw = max(0.0, min(x2, kx2) - max(x1, kx1))
            ih = max(0.0, min(y2, ky2) - max(y1, ky1))
            inter = iw * ih
            k_area = (kx2 - kx1) * (ky2 - ky1)
            union = area + k_area - inter
            if inter / (union + 1e-9) > iou_thr or inter / (min(area, k_area) + 1e-9) > 0.85:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
    return kept


class ClassicTargetDetector(Detector):
    name = "classic"

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.cfg.clahe:
            gray = self._clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = auto_canny(gray)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = gray.shape[:2]
        dets: list[Detection] = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri < 4 * self.cfg.min_side:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if not 4 <= len(approx) <= 6:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w > 0.9 * w_img or h > 0.9 * h_img:  # frame borders / horizon artefacts
                continue
            # Shape measured on the minimum-area rotated rectangle, so a square board is
            # recognised at any rotation while elongated shapes (vehicles, roofs) are not,
            # whatever their heading.
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            short, long_ = min(rw, rh), max(rw, rh)
            if short < self.cfg.min_side:
                continue
            aspect = long_ / max(short, 1e-6)
            if aspect > max(self.cfg.aspect_max, 1.0 / self.cfg.aspect_min):
                continue
            area = cv2.contourArea(c)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < self.cfg.min_solidity:
                continue
            rectangularity = area / max(rw * rh, 1e-6)
            if rectangularity < self.cfg.min_rectangularity:  # rejects discs, blobs, crosses
                continue
            # Score: how close to an ideal filled square the contour is.
            squareness = 1.0 - min(1.0, (aspect - 1.0) / 0.25)
            vertex_bonus = 1.0 if len(approx) == 4 else 0.85
            score = float(np.clip(0.4 * solidity + 0.4 * rectangularity + 0.2 * squareness, 0, 1))
            score *= vertex_bonus
            dets.append(Detection((float(x), float(y), float(x + w), float(y + h)), score, 0, "target"))
        return nms_contained(dets)
