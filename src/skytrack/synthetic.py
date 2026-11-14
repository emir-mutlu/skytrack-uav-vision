"""Deterministic synthetic aerial scene generator.

Produces textured 'terrain' with moving square targets, non-target distractors
(circles, elongated rectangles), camera jitter, sensor noise and occasional
occlusion. Every frame comes with ground-truth boxes so the pipeline can be
demoed and evaluated without real flight footage.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from .types import BBox


@dataclass
class _Mover:
    x: float
    y: float
    vx: float
    vy: float
    size: int
    kind: str  # "target" | "circle" | "bar"


def _terrain(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    layers = np.zeros((h, w), np.float32)
    for scale, amp in ((8, 0.5), (32, 0.3), (128, 0.2)):
        small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
        layers += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    layers = cv2.normalize(layers, None, 0, 1, cv2.NORM_MINMAX)
    green = np.stack([60 + 60 * layers, 110 + 70 * layers, 70 + 50 * layers], axis=-1)
    return np.clip(green, 0, 255).astype(np.uint8)


def _draw_target(img: np.ndarray, x: int, y: int, s: int) -> None:
    cv2.rectangle(img, (x, y), (x + s, y + s), (20, 20, 20), -1)
    m = max(2, s // 5)
    cv2.rectangle(img, (x + m, y + m), (x + s - m, y + s - m), (235, 235, 235), -1)
    m2 = max(3, s // 3)
    cv2.rectangle(img, (x + m2, y + m2), (x + s - m2, y + s - m2), (20, 20, 200), -1)


class SyntheticScene:
    def __init__(
        self,
        width: int = 960,
        height: int = 540,
        n_targets: int = 2,
        n_distractors: int = 3,
        seed: int = 7,
        noise: float = 6.0,
        jitter: float = 2.0,
        occlusion: bool = True,
    ):
        self.w, self.h = width, height
        self.rng = np.random.default_rng(seed)
        self.noise, self.jitter, self.occlusion = noise, jitter, occlusion
        pad = 80
        self.bg = _terrain(width + 2 * pad, height + 2 * pad, self.rng)
        self.pad = pad
        self.movers: list[_Mover] = []
        for i in range(n_targets):
            self.movers.append(self._spawn("target", 40 + 12 * i))
        for i in range(n_distractors):
            self.movers.append(self._spawn(("circle", "bar")[i % 2], 30 + 6 * i))
        self.t = 0

    def _spawn(self, kind: str, size: int) -> _Mover:
        r = self.rng
        return _Mover(
            x=float(r.uniform(size, self.w - 2 * size)),
            y=float(r.uniform(size, self.h - 2 * size)),
            vx=float(r.uniform(-3, 3)),
            vy=float(r.uniform(-2, 2)),
            size=size,
            kind=kind,
        )

    def step(self) -> tuple[np.ndarray, list[BBox]]:
        ox = int(self.pad + self.rng.normal(0, self.jitter) + 20 * np.sin(self.t / 60))
        oy = int(self.pad + self.rng.normal(0, self.jitter))
        frame = self.bg[oy : oy + self.h, ox : ox + self.w].copy()
        gt: list[BBox] = []
        for m in self.movers:
            m.x += m.vx
            m.y += m.vy
            if not 0 <= m.x <= self.w - m.size:
                m.vx *= -1
                m.x = float(np.clip(m.x, 0, self.w - m.size))
            if not 0 <= m.y <= self.h - m.size:
                m.vy *= -1
                m.y = float(np.clip(m.y, 0, self.h - m.size))
            x, y, s = int(m.x), int(m.y), m.size
            if m.kind == "target":
                _draw_target(frame, x, y, s)
                gt.append((float(x), float(y), float(x + s), float(y + s)))
            elif m.kind == "circle":
                cv2.circle(frame, (x + s // 2, y + s // 2), s // 2, (30, 30, 30), -1)
            else:
                cv2.rectangle(frame, (x, y), (x + 3 * s, y + s // 3), (40, 40, 40), -1)
        if self.occlusion and (self.t // 90) % 4 == 3:  # periodic cloud/haze band
            band = frame[:, self.w // 3 : self.w // 3 + 120]
            frame[:, self.w // 3 : self.w // 3 + 120] = cv2.addWeighted(
                band, 0.15, np.full_like(band, 225), 0.85, 0
            )
        if self.noise > 0:
            n = self.rng.normal(0, self.noise, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8)
        self.t += 1
        return frame, gt

    def frames(self, n: int) -> Iterator[tuple[np.ndarray, list[BBox]]]:
        for _ in range(n):
            yield self.step()
