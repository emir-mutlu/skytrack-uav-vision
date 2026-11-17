"""Deterministic synthetic aerial scene generator.

Renders a procedural top-down countryside - meadows, forest canopy, a pond, roads with
traffic and buildings with cast shadows - seen through a UAV camera with optics blur,
vignetting, sensor noise and camera shake. Square ground-target boards (what the classic
detector looks for) drift and rotate across it, cars drive along the roads as realistic
distractors, and a cloud periodically drifts through the view to test occlusion handling.

Everything is drawn with sub-pixel, anti-aliased geometry, so motion is smooth instead of
stepping pixel by pixel. Every frame comes with ground-truth boxes, so the pipeline can be
demoed and evaluated without flight footage.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from .types import BBox

SUN = np.array([3.0, 4.0])  # shadow offset in px: light from the upper left
SHIFT = 4  # fixed-point bits for sub-pixel drawing
ONE = 1 << SHIFT
AA = cv2.LINE_AA


def _rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    return (b, g, r)


# ----------------------------------------------------------------------------- noise
def _smooth_noise(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    sh, sw = int(h / scale) + 3, int(w / scale) + 3
    small = cv2.GaussianBlur(rng.random((sh, sw)).astype(np.float32), (0, 0), 0.7)
    big = cv2.resize(small, (int(sw * scale), int(sh * scale)), interpolation=cv2.INTER_CUBIC)
    oy = int(rng.integers(0, max(1, big.shape[0] - h)))
    ox = int(rng.integers(0, max(1, big.shape[1] - w)))
    return big[oy:oy + h, ox:ox + w]


def _fbm(h: int, w: int, rng: np.random.Generator, base: float, octaves: int = 6,
         gain: float = 0.55) -> np.ndarray:
    acc = np.zeros((h, w), np.float32)
    amp, total, scale = 1.0, 0.0, base
    for _ in range(octaves):
        acc += amp * _smooth_noise(h, w, max(scale, 2.0), rng)
        total += amp
        amp *= gain
        scale /= 2.0
    return cv2.normalize(acc / total, None, 0.0, 1.0, cv2.NORM_MINMAX)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------- geometry
def _rect(cx: float, cy: float, length: float, width: float, ang: float, chamfer: float = 0.0) -> np.ndarray:
    """Corners of a (chamfered) rectangle whose long axis points along `ang` (radians)."""
    hl, hw = length / 2.0, width / 2.0
    if chamfer > 0:
        c = min(chamfer, hw * 0.9)
        pts = np.array([[-hl + c, -hw], [hl - c, -hw], [hl, -hw + c], [hl, hw - c],
                        [hl - c, hw], [-hl + c, hw], [-hl, hw - c], [-hl, -hw + c]])
    else:
        pts = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    ca, sa = math.cos(ang), math.sin(ang)
    return pts @ np.array([[ca, sa], [-sa, ca]]) + (cx, cy)


def _fill(img: np.ndarray, pts: np.ndarray, color) -> None:
    cv2.fillConvexPoly(img, np.round(pts * ONE).astype(np.int32), color, AA, SHIFT)


def _shade(img: np.ndarray, pts: np.ndarray, alpha: float) -> None:
    """Darken a polygon area (soft cast shadow)."""
    x1, y1 = np.floor(pts.min(0)).astype(int) - 2
    x2, y2 = np.ceil(pts.max(0)).astype(int) + 3
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round((pts - (x1, y1)) * ONE).astype(np.int32), 255, AA, SHIFT)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.9)
    roi[:] = (roi.astype(np.float32) * (1.0 - alpha * mask[..., None] / 255.0)).astype(np.uint8)


def _bezier(p0, p1, p2, p3, n: int = 700) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# ----------------------------------------------------------------------------- terrain
def _terrain(w: int, h: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """Procedural countryside. Returns (BGR image, road centre-lines)."""
    cover, moist, soil_n = _fbm(h, w, rng, 420), _fbm(h, w, rng, 300), _fbm(h, w, rng, 170)
    grain = _fbm(h, w, rng, 12, octaves=3)
    forest = _smoothstep(0.60, 0.68, cover)
    water = _smoothstep(0.90, 0.915, moist) * (1 - forest)
    dry = _smoothstep(0.50, 0.64, moist) * (1 - forest) * (1 - water)
    soil = _smoothstep(0.76, 0.81, soil_n) * (1 - forest) * (1 - water) * 0.8

    img = np.empty((h, w, 3), np.float32)
    lo, hi = np.array(_rgb(98, 132, 68), np.float32), np.array(_rgb(126, 152, 82), np.float32)
    img[:] = lo + (hi - lo) * cover[..., None]
    for mask, rgb in ((dry, (158, 152, 100)), (soil, (134, 112, 86)), (water, (52, 78, 76)),
                      (forest, (44, 70, 40))):
        img += (np.array(_rgb(*rgb), np.float32) - img) * mask[..., None]
    img *= (0.9 + 0.2 * grain)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # tree crowns: dense in forest, sparse over meadows; shadow, crown, sunlit highlight
    n = int(w * h / 260)
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    f = forest[ys.astype(int).clip(0, h - 1), xs.astype(int).clip(0, w - 1)]
    keep = (rng.random(n) < f * 0.9) | (rng.random(n) < 0.004)
    keep &= water[ys.astype(int).clip(0, h - 1), xs.astype(int).clip(0, w - 1)] < 0.2
    for x, y in zip(xs[keep], ys[keep], strict=True):
        r = rng.uniform(3.5, 7.0)
        g = int(rng.integers(64, 98))
        shadow = (int((x + 2.2) * ONE), int((y + 2.8) * ONE))
        cv2.circle(img, shadow, int(r * ONE), _rgb(26, 40, 24), -1, AA, SHIFT)
        cv2.circle(img, (int(x * ONE), int(y * ONE)), int(r * ONE), _rgb(g - 30, g, g - 40), -1, AA, SHIFT)
        cv2.circle(img, (int((x - r * 0.3) * ONE), int((y - r * 0.3) * ONE)), int(r * 0.5 * ONE),
                   _rgb(g - 14, g + 18, g - 26), -1, AA, SHIFT)

    # a dirt track, then two asphalt roads with gravel shoulders and dashed centre lines
    track = _bezier((-40, rng.uniform(0.1, 0.9) * h), (rng.uniform(0, w), rng.uniform(0, h)),
                    (rng.uniform(0, w), rng.uniform(0, h)), (w + 40, rng.uniform(0.1, 0.9) * h))
    cv2.polylines(img, [np.round(track * ONE).astype(np.int32)], False, _rgb(146, 124, 94), 5, AA, SHIFT)
    roads = [
        _bezier((-60, rng.uniform(0.25, 0.75) * h), (w * 0.33, rng.uniform(0, h)),
                (w * 0.66, rng.uniform(0, h)), (w + 60, rng.uniform(0.25, 0.75) * h)),
        _bezier((rng.uniform(0.25, 0.75) * w, -60), (rng.uniform(0, w), h * 0.33),
                (rng.uniform(0, w), h * 0.66), (rng.uniform(0.25, 0.75) * w, h + 60)),
    ]
    for pts in roads:
        p = np.round(pts * ONE).astype(np.int32)
        cv2.polylines(img, [p], False, _rgb(150, 142, 120), 22, AA, SHIFT)
        cv2.polylines(img, [p], False, _rgb(84, 87, 91), 16, AA, SHIFT)
        for i in range(0, len(p) - 5, 12):
            cv2.polylines(img, [p[i:i + 5]], False, _rgb(214, 208, 172), 1, AA, SHIFT)

    # buildings beside the first road: elongated roofs aligned with the road
    road = roads[0]
    for i in rng.choice(np.arange(40, len(road) - 40), size=9, replace=False):
        d = road[i + 3] - road[i - 3]
        ang = math.atan2(d[1], d[0]) + math.radians(rng.uniform(-6, 6))
        normal = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        length = rng.uniform(24, 40)
        width = length / rng.uniform(1.8, 2.5)
        c = road[i] + normal * rng.choice([-1, 1]) * (24 + width / 2 + rng.uniform(0, 12))
        body = _rect(c[0], c[1], length, width, ang)
        _shade(img, body + SUN * 1.7, 0.55)
        roof = [(170, 86, 64), (126, 128, 134), (188, 176, 150)][int(rng.integers(3))]
        _fill(img, body, _rgb(*roof))
        half = _rect(c[0], c[1] + 0, length, width / 2, ang)
        _fill(img, half + normal * width / 4, _rgb(*(int(v * 0.8) for v in roof)))  # shaded roof half
    return img, roads


# ----------------------------------------------------------------------------- objects
@dataclass
class _Mover:
    x: float  # centre, world px
    y: float
    vx: float = 0.0
    vy: float = 0.0
    size: float = 40.0  # board side or car length
    kind: str = "target"  # "target" | "car"
    angle: float = 0.0  # radians
    spin: float = 0.0  # radians / frame
    road: int = -1
    s: float = 0.0  # arc length along the road
    speed: float = 0.0
    color: tuple = field(default_factory=lambda: _rgb(220, 220, 224))


def _board_corners(m: _Mover, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return _rect(m.x - dx, m.y - dy, m.size * scale, m.size * scale, m.angle)


def _draw_board(img: np.ndarray, m: _Mover, dx: float, dy: float) -> None:
    _shade(img, _board_corners(m, dx, dy) + SUN, 0.42)
    for scale, rgb in ((1.0, (236, 236, 230)), (0.84, (28, 28, 30)), (0.56, (236, 236, 230)),
                       (0.30, (206, 38, 42))):
        _fill(img, _board_corners(m, dx, dy, scale), _rgb(*rgb))


def _draw_car(img: np.ndarray, m: _Mover, dx: float, dy: float) -> None:
    cx, cy, length, ang = m.x - dx, m.y - dy, m.size, m.angle
    width = length * 0.46
    ca, sa = math.cos(ang), math.sin(ang)
    _shade(img, _rect(cx, cy, length, width, ang, 2.5) + SUN * 0.8, 0.5)
    _fill(img, _rect(cx, cy, length, width, ang, 2.5), m.color)
    glass = _rgb(38, 46, 56)
    _fill(img, _rect(cx + ca * length * 0.17, cy + sa * length * 0.17, length * 0.17, width * 0.8, ang), glass)
    _fill(img, _rect(cx - ca * length * 0.31, cy - sa * length * 0.31, length * 0.11, width * 0.74, ang), glass)


CAR_COLORS = [(232, 232, 236), (190, 194, 200), (40, 42, 48), (164, 36, 40), (38, 72, 140), (210, 190, 150)]


class _World:
    """Static terrain plus the moving objects that live on it."""

    def __init__(self, width: int, height: int, seed: int, n_targets: int, n_cars: int,
                 bounds: tuple[float, float, float, float], occlusion: bool):
        self.rng = np.random.default_rng(seed)
        self.w, self.h = width, height
        self.terrain, self.roads = _terrain(width, height, self.rng)
        self.road_len = [np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
                         for r in self.roads]
        self.bounds = bounds
        self.occlusion = occlusion
        self.t = 0
        r = self.rng
        bx0, by0, bx1, by1 = bounds
        self.movers: list[_Mover] = []
        for i in range(n_targets):
            size = 40.0 + 12.0 * i
            vx, vy = r.uniform(-3, 3), r.uniform(-2, 2)
            if math.hypot(vx, vy) < 0.8:
                vx = math.copysign(1.2, vx)
            self.movers.append(_Mover(
                x=r.uniform(bx0 + size, bx1 - size), y=r.uniform(by0 + size, by1 - size), vx=vx, vy=vy,
                size=size, kind="target", angle=r.uniform(0, math.pi / 2), spin=r.uniform(-0.006, 0.006)))
        for i in range(n_cars):
            road = i % len(self.roads)
            self.movers.append(_Mover(
                x=0, y=0, size=r.uniform(18, 24), kind="car", road=road,
                s=r.uniform(0, self.road_len[road][-1]),
                speed=r.choice([-1, 1]) * r.uniform(0.9, 1.8),
                color=_rgb(*CAR_COLORS[int(r.integers(len(CAR_COLORS)))])))
        self._place_cars()
        self._cloud = self._make_cloud(int(0.32 * min(bx1 - bx0, by1 - by0) + 60))

    def _make_cloud(self, radius: int) -> np.ndarray:
        size = 2 * radius
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.hypot(xx - radius, yy - radius) / radius
        puff = _fbm(size, size, self.rng, size / 3, octaves=4)
        return (np.clip(1.15 - d - 0.45 * (1 - puff), 0, 1) ** 1.3 * 0.9).astype(np.float32)

    def _place_cars(self) -> None:
        for m in self.movers:
            if m.kind != "car":
                continue
            path, cum = self.roads[m.road], self.road_len[m.road]
            m.s %= cum[-1]
            x, y = np.interp(m.s, cum, path[:, 0]), np.interp(m.s, cum, path[:, 1])
            ahead = (m.s + 4.0) % cum[-1]
            hx, hy = np.interp(ahead, cum, path[:, 0]) - x, np.interp(ahead, cum, path[:, 1]) - y
            norm = math.hypot(hx, hy) + 1e-9
            lane = 3.6 if m.speed > 0 else -3.6  # drive on the right-hand side
            m.x, m.y = x - hy / norm * lane, y + hx / norm * lane
            m.angle = math.atan2(hy, hx) + (0.0 if m.speed > 0 else math.pi)

    def update(self) -> None:
        bx0, by0, bx1, by1 = self.bounds
        for m in self.movers:
            if m.kind == "target":
                m.x += m.vx
                m.y += m.vy
                m.angle += m.spin
                half = m.size * 0.72  # rotated board fits in this radius
                if not bx0 + half <= m.x <= bx1 - half:
                    m.vx *= -1
                    m.x = float(np.clip(m.x, bx0 + half, bx1 - half))
                if not by0 + half <= m.y <= by1 - half:
                    m.vy *= -1
                    m.y = float(np.clip(m.y, by0 + half, by1 - half))
            else:
                m.s += m.speed
        self._place_cars()
        self.t += 1

    def cloud_center(self) -> tuple[float, float] | None:
        """A cloud drifts across the target area during frames 270-359 of every 360."""
        phase = self.t % 360
        if not self.occlusion or phase < 270:
            return None
        bx0, by0, bx1, by1 = self.bounds
        radius = self._cloud.shape[0] / 2
        p = (phase - 270) / 90.0
        return bx0 - radius + p * (bx1 - bx0 + 2 * radius), (by0 + by1) / 2

    def render(self, x0: float, y0: float, w: int, h: int) -> np.ndarray:
        """View of size (w, h) whose top-left corner is at world (x0, y0); sub-pixel exact."""
        ix, iy = int(math.floor(x0)), int(math.floor(y0))
        fx, fy = x0 - ix, y0 - iy
        ix, iy = int(np.clip(ix, 0, self.w - w - 1)), int(np.clip(iy, 0, self.h - h - 1))
        view = self.terrain[iy:iy + h + 1, ix:ix + w + 1].copy()
        for m in self.movers:
            if (m.x - ix < -m.size or m.x - ix > w + m.size or m.y - iy < -m.size or m.y - iy > h + m.size):
                continue
            (_draw_car if m.kind == "car" else _draw_board)(view, m, ix, iy)
        cc = self.cloud_center()
        if cc is not None:
            r = self._cloud.shape[0] // 2
            cx, cy = int(cc[0]) - ix, int(cc[1]) - iy
            x1, y1, x2, y2 = max(0, cx - r), max(0, cy - r), min(view.shape[1], cx + r), min(view.shape[0], cy + r)
            if x2 > x1 and y2 > y1:
                a = self._cloud[y1 - cy + r:y2 - cy + r, x1 - cx + r:x2 - cx + r, None]
                roi = view[y1:y2, x1:x2].astype(np.float32)
                view[y1:y2, x1:x2] = (roi + (244.0 - roi) * a).astype(np.uint8)
        if fx or fy:
            view = cv2.warpAffine(view, np.float32([[1, 0, -fx], [0, 1, -fy]]), (w, h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return view[:h, :w]

    def gt(self, x0: float, y0: float, w: int, h: int) -> list[BBox]:
        boxes: list[BBox] = []
        for m in self.movers:
            if m.kind != "target":
                continue
            pts = _board_corners(m, x0, y0)
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.5 * (x2 - x1) * (y2 - y1):
                continue  # mostly outside the view
            boxes.append((float(cx1), float(cy1), float(cx2), float(cy2)))
        return boxes


class _Optics:
    """Lens blur, vignetting and sensor noise (from a pre-generated noise bank)."""

    def __init__(self, w: int, h: int, noise: float, rng: np.random.Generator):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        vig = np.round(255.0 * (1.0 - 0.13 * r2)).astype(np.uint8)
        self.vignette = cv2.merge([vig, vig, vig])
        self.noise: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(5 if noise > 0 else 0):  # signed noise split in two saturating uint8 parts
            n = rng.normal(0, noise, (h, w, 3))
            self.noise.append((np.clip(n, 0, 255).astype(np.uint8), np.clip(-n, 0, 255).astype(np.uint8)))
        self.rng = rng

    def __call__(self, view: np.ndarray) -> np.ndarray:
        v = cv2.multiply(cv2.GaussianBlur(view, (0, 0), 0.6), self.vignette, scale=1.0 / 255.0)
        if self.noise:
            pos, neg = self.noise[int(self.rng.integers(len(self.noise)))]
            v = cv2.subtract(cv2.add(v, pos), neg)
        return v


# ----------------------------------------------------------------------------- cameras
class SyntheticScene:
    """Fixed, slightly shaking camera over the scene; targets stay inside the view."""

    def __init__(
        self,
        width: int = 960,
        height: int = 540,
        n_targets: int = 2,
        n_distractors: int = 3,
        seed: int = 7,
        noise: float = 3.0,
        jitter: float = 1.2,
        occlusion: bool = True,
    ):
        self.w, self.h = width, height
        self.pad = 60
        self.world = _World(width + 2 * self.pad, height + 2 * self.pad, seed, n_targets, n_distractors,
                            (self.pad, self.pad, self.pad + width, self.pad + height), occlusion)
        self.rng = np.random.default_rng(seed + 101)
        self.jitter = jitter
        self.optics = _Optics(width, height, noise, self.rng)
        self.t = 0

    @property
    def movers(self) -> list[_Mover]:
        return self.world.movers

    def step(self) -> tuple[np.ndarray, list[BBox]]:
        self.world.update()
        x0 = self.pad + 18 * math.sin(self.t / 60) + self.rng.normal(0, self.jitter)
        y0 = self.pad + 10 * math.sin(self.t / 97) + self.rng.normal(0, self.jitter)
        frame = self.optics(self.world.render(x0, y0, self.w, self.h))
        self.t += 1
        return frame, self.world.gt(x0, y0, self.w, self.h)

    def frames(self, n: int) -> Iterator[tuple[np.ndarray, list[BBox]]]:
        for _ in range(n):
            yield self.step()


class PanningScene:
    """The synthetic world seen through a pan/tilt camera.

    The view is a window into a larger world whose position follows the gimbal angles
    (pixels per degree from the lens HFOV). Feeding it the simulated gimbal attitude
    closes the loop - detector -> tracker -> lock -> gimbal command -> camera motion -
    which is how gimbal-follow is demoed and regression-tested without hardware.
    """

    def __init__(self, width: int = 960, height: int = 540, hfov_deg: float = 80.0,
                 margin: float = 0.35, seed: int = 7, noise: float = 3.0, occlusion: bool = True,
                 n_targets: int = 2, n_distractors: int = 4):
        self.w, self.h = width, height
        self.mw, self.mh = int(width * margin), int(height * margin)
        ww, wh = width + 2 * self.mw, height + 2 * self.mh
        self.world = _World(ww, wh, seed, n_targets, n_distractors, (8, 8, ww - 8, wh - 8), occlusion)
        self.rng = np.random.default_rng(seed + 1)
        self.optics = _Optics(width, height, noise, self.rng)
        self.ppd = width / hfov_deg
        self._ref: tuple[float, float] | None = None
        for m in self.world.movers:  # start every target inside the initial field of view
            if m.kind == "target":
                m.x = float(self.rng.uniform(self.mw + 0.25 * width, self.mw + 0.75 * width))
                m.y = float(self.rng.uniform(self.mh + 0.25 * height, self.mh + 0.75 * height))

    def gimbal_limits(self, ref_pitch_deg: float = -45.0, ref_yaw_deg: float = 0.0):
        """Gimbal (pitch, yaw) ranges the view can follow - give these to the simulated
        gimbal so it stops at its end stops instead of turning past the rendered world."""
        dp, dy = (self.mh - 2) / self.ppd, (self.mw - 2) / self.ppd
        return (ref_pitch_deg - dp, ref_pitch_deg + dp), (ref_yaw_deg - dy, ref_yaw_deg + dy)

    def step(self, gimbal_pitch_deg: float, gimbal_yaw_deg: float) -> tuple[np.ndarray, list[BBox]]:
        self.world.update()
        if self._ref is None:
            self._ref = (gimbal_pitch_deg, gimbal_yaw_deg)
        dx = float(np.clip((gimbal_yaw_deg - self._ref[1]) * self.ppd, -self.mw + 1, self.mw - 2))
        dy = float(np.clip(-(gimbal_pitch_deg - self._ref[0]) * self.ppd, -self.mh + 1, self.mh - 2))
        x0, y0 = self.mw + dx, self.mh + dy
        view = self.optics(self.world.render(x0, y0, self.w, self.h))
        return view, self.world.gt(x0, y0, self.w, self.h)
