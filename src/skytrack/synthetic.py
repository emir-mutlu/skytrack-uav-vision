"""Deterministic synthetic aerial scene generator.

The ground is a real aerial photograph by default: Dutch national orthophotography of farmland
in the Gelderse Vallei (Luchtfoto Actueel Ortho 8 cm RGB, Beeldmateriaal Nederland, via PDOK,
open data CC BY 4.0), resampled to 0.2 m per pixel by scripts/build_aerial.py - meadows with
mowing patterns, tree rows and hedges with their shadows, farm buildings with solar panels,
parked vehicles, lanes and a creek. Without the photo (or with background="procedural") the
countryside is drawn procedurally to the same scale instead, with roads and moving cars.

A UAV camera sees it with optics blur, vignetting, sensor noise and camera shake. Square
ground-target boards (what the classic detector looks for) drift and rotate across it, their
shadows cast by the photo's own sun, and a cloud periodically drifts through the view to test
occlusion handling.

Everything is drawn with sub-pixel, anti-aliased geometry, so motion is smooth instead of
stepping pixel by pixel. Every frame comes with ground-truth boxes, so the pipeline can be
demoed and evaluated without flight footage.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .types import BBox

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
# The world is drawn to scale, as an orthophoto would show it: GSD_M metres per pixel, one
# sun for every shading and shadow. At 0.2 m/px a family house is ~60 px long, a car ~22 px,
# a two-lane road 35 px wide and a mature tree crown 25-50 px across.
GSD_M = 0.2
SUN_ELEV = math.radians(62.0)
SUN_DIR = np.array([0.6, 0.8])  # horizontal direction shadows fall in: light from the upper left
LIGHT = np.array([-SUN_DIR[0] * math.cos(SUN_ELEV), -SUN_DIR[1] * math.cos(SUN_ELEV), math.sin(SUN_ELEV)])
SUN = SUN_DIR * 3.0  # shadow offset of the flat target boards, px

_CROPS = (  # RGB, row period px, row contrast, tractor tramlines
    ((104, 122, 70), 0.0, 0.00, False),  # pasture
    ((96, 126, 64), 3.6, 0.07, True),  # young cereal
    ((168, 152, 100), 3.6, 0.06, True),  # ripe cereal
    ((158, 146, 114), 6.0, 0.08, True),  # stubble after harvest
    ((118, 98, 76), 4.5, 0.15, False),  # ploughed
    ((124, 140, 84), 24.0, 0.05, False),  # mown hay, mower swaths
)
_ROOFS = {  # RGB, texture period px, contrast, ribs run down the slope (sheet metal) or tiles along it
    "tile": ((150, 84, 62), 2.4, 0.08, False),
    "slate": ((86, 88, 94), 2.4, 0.07, False),
    "brown": ((116, 92, 76), 2.4, 0.07, False),
    "metal": ((150, 152, 154), 3.2, 0.09, True),
    "green": ((76, 94, 78), 3.2, 0.09, True),
    "rust": ((124, 70, 56), 3.2, 0.09, True),
}


def _cast(height_m: float) -> np.ndarray:
    """Shadow offset in world px of something `height_m` above the ground."""
    return SUN_DIR * (height_m / math.tan(SUN_ELEV) / GSD_M)


def _paint(img: np.ndarray, poly: np.ndarray, colour, alpha: float = 1.0) -> None:
    """Anti-aliased convex polygon on a float image; `colour` is an RGB triple or a function
    of the pixel coordinates (X, Y) returning an (h, w, 3) array."""
    x0, y0 = np.floor(poly.min(0)).astype(int) - 1
    x1, y1 = np.ceil(poly.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, np.round((poly - (x0, y0)) * ONE).astype(np.int32), 255, AA, SHIFT)
    m = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    if callable(colour):
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        col = colour(xx, yy)
    else:
        col = np.asarray(colour, np.float32)
    roi = img[y0:y1, x0:x1]
    roi += (col - roi) * m


def _line(img: np.ndarray, p: np.ndarray, q: np.ndarray, width: float, colour, alpha: float) -> None:
    d, mid = q - p, (p + q) / 2
    _paint(img, _rect(mid[0], mid[1], float(np.linalg.norm(d)) + 1.0, width, math.atan2(d[1], d[0])),
           colour, alpha)


def _stroke(shape: tuple[int, int], lines: list[np.ndarray], width: float) -> np.ndarray:
    """Coverage (0-1) of anti-aliased polylines of the given width, all drawn into one mask."""
    m = np.zeros(shape, np.uint8)
    for pts in lines:
        cv2.polylines(m, [np.round(pts * ONE).astype(np.int32)], False, 255, max(1, round(width)), AA, SHIFT)
    return m.astype(np.float32) / 255.0


def _offset(pts: np.ndarray, d: float) -> np.ndarray:
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return pts + np.stack([-t[:, 1], t[:, 0]], 1) * d


def _fields(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """Farmland: long parcels on a slightly rotated grid, each with its own crop, row direction,
    tractor tramlines and a grass margin. Returns (RGB float image, hedgerow lines)."""
    ang = rng.uniform(-0.45, 0.45)
    ca, sa = math.cos(ang), math.sin(ang)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx * ca + yy * sa, yy * ca - xx * sa
    out = np.empty((h, w, 3), np.float32)
    edge = np.full((h, w), 1e9, np.float32)  # distance to the nearest parcel boundary
    crops = np.array([c[0] for c in _CROPS], np.float32)
    vb = [float(v.min()) - rng.uniform(0, 150)]
    while vb[-1] < float(v.max()):
        vb.append(vb[-1] + rng.uniform(150, 260))
    sv = np.searchsorted(np.asarray(vb), v)
    hedges: list[np.ndarray] = []
    for i in range(1, len(vb)):
        m = sv == i
        if not m.any():
            continue
        uu, vv = u[m], v[m]
        ub = [float(uu.min()) - rng.uniform(0, 300)]
        while ub[-1] < float(uu.max()):
            ub.append(ub[-1] + (vb[i] - vb[i - 1]) * rng.uniform(1.7, 3.4))
        uba = np.asarray(ub)
        su = np.searchsorted(uba, uu)
        k = len(uba) + 1
        kind = rng.integers(0, len(_CROPS), k)
        along = rng.random(k) < 0.8  # rows mostly run the length of the parcel
        period = np.array([_CROPS[c][1] for c in kind], np.float32)[su]
        contrast = np.array([_CROPS[c][2] for c in kind], np.float32)[su]
        tram = np.array([_CROPS[c][3] for c in kind])[su]
        tone = rng.uniform(0.93, 1.07, k).astype(np.float32)[su]
        across = np.where(along[su], vv, uu)
        rows = np.where(period > 0, np.sin(2 * np.pi * across / np.maximum(period, 1e-3)), 0.0)
        col = crops[kind[su]] * (tone * (1 + contrast * rows))[:, None]
        tl = np.mod(across + rng.uniform(0, 104), 104.0)  # wheel tracks 1.5 m apart every ~21 m
        col *= np.where(tram & ((np.abs(tl - 2.0) < 1.1) | (np.abs(tl - 9.5) < 1.1)), 0.9, 1.0)[:, None]
        out[m] = col
        lo, hi = uba[np.clip(su - 1, 0, len(uba) - 1)], uba[np.clip(su, 0, len(uba) - 1)]
        edge[m] = np.minimum(np.minimum(vv - vb[i - 1], vb[i] - vv), np.minimum(uu - lo, hi - uu))
        if rng.random() < 0.35:  # a hedgerow along part of this parcel row's boundary
            a = rng.uniform(float(uu.min()), float(uu.max()))
            us = np.arange(a, a + rng.uniform(300, 900), 12.0)
            hedges.append(np.stack([us * ca - vb[i - 1] * sa, us * sa + vb[i - 1] * ca], 1))
    margin = 1.0 - _smoothstep(1.5, 4.0, edge)  # grass strip along every boundary
    out += (np.array((112, 120, 76), np.float32) - out) * (margin * 0.85)[..., None]
    return out, hedges


@dataclass
class _Building:
    c: np.ndarray
    length: float
    width: float
    ang: float
    material: str
    hip: bool
    pitch: float  # radians
    height_m: float
    house: bool
    road_pt: np.ndarray  # nearest road centre point, for the driveway
    chimney: bool = False
    solar: bool = False


def _plan_buildings(roads: list[np.ndarray], forest: np.ndarray, w: int, h: int,
                    rng: np.random.Generator) -> list[_Building]:
    """Houses and farm buildings set back from the roads, never in woodland or on each other."""
    want = max(3, int(w * h / 180_000))
    allpts = np.concatenate(roads)
    out: list[_Building] = []
    for _ in range(300):
        if len(out) >= want:
            break
        rd = roads[0] if rng.random() < 0.65 else roads[int(rng.integers(len(roads)))]
        i = int(rng.integers(30, len(rd) - 30))
        d = rd[i + 4] - rd[i - 4]
        d /= np.linalg.norm(d) + 1e-9
        normal = np.array([-d[1], d[0]]) * rng.choice([-1, 1])
        if rng.random() < 0.25:  # barn: long, low-pitched sheet-metal roof
            length, width = rng.uniform(95, 130), rng.uniform(52, 66)
            material = str(rng.choice(["metal", "green", "rust"]))
            hip, pitch, height, house = False, 18.0, 7.5, False
        else:  # family house, clearly longer than wide
            length = rng.uniform(55, 70)
            width = length / rng.uniform(1.65, 1.95)
            material = str(rng.choice(["tile", "tile", "slate", "brown"]))
            hip, pitch, height, house = bool(rng.random() < 0.45), 35.0, 6.5, True
        setback = 23.5 + rng.uniform(45, 85) + width / 2
        c = rd[i] + normal * setback
        ang = math.atan2(d[1], d[0]) + rng.uniform(-0.05, 0.05)
        reach = math.hypot(length, width) / 2
        if not (reach + 50 < c[0] < w - reach - 50 and reach + 50 < c[1] < h - reach - 50):
            continue
        if forest[int(c[1]), int(c[0])] > 0.15:
            continue
        if np.min(np.linalg.norm(allpts - c, axis=1)) < setback - 6:
            continue
        if any(np.linalg.norm(c - o.c) < reach + math.hypot(o.length, o.width) / 2 + 45 for o in out):
            continue
        out.append(_Building(c, length, width, ang, material, hip, math.radians(pitch), height, house, rd[i],
                             chimney=house and rng.random() < 0.6, solar=house and rng.random() < 0.3))
    return out


def _yard(img: np.ndarray, b: _Building, rng: np.random.Generator) -> None:
    """Front lawn and a driveway to the road for houses, a concrete apron for barns."""
    ax = np.array([math.cos(b.ang), math.sin(b.ang)])
    to_road = b.road_pt - b.c
    to_road /= np.linalg.norm(to_road) + 1e-9
    if b.house:
        c = b.c + to_road * 10
        phase = rng.uniform(0, 10)
        mottle = cv2.GaussianBlur(rng.random((64, 64)).astype(np.float32), (0, 0), 3)

        def lawn(xx, yy):
            s = (xx - c[0]) * ax[0] + (yy - c[1]) * ax[1]
            g = cv2.resize(mottle, (xx.shape[1], xx.shape[0]), interpolation=cv2.INTER_LINEAR)
            return np.array((112, 132, 76), np.float32) * ((1 + 0.035 * np.sin((s + phase) * 0.6))
                                                          * (0.86 + 0.28 * g))[..., None]

        _paint(img, _rect(c[0], c[1], b.length + 76, b.width + 30, b.ang), lawn, 0.55)
    else:
        c = b.c + to_road * (b.width / 2 + 22)
        _paint(img, _rect(c[0], c[1], b.length * 0.9, 44, b.ang), (150, 148, 140), 0.85)
    start = b.c + ax * b.length * rng.uniform(-0.3, 0.3) + to_road * b.width / 2
    end = b.road_pt + ax * float(np.dot(start - b.road_pt, ax)) * 0.8 - to_road * 20
    _line(img, start, end, 15, (156, 152, 142), 0.9)


def _roof(img: np.ndarray, b: _Building, rng: np.random.Generator) -> None:
    """Gable or hip roof: each plane lit by its own normal, with tile rows or sheet ribs,
    weathering, a crisp ridge and eave line, and the odd chimney or solar array."""
    rgb, period, contrast, ribs = _ROOFS[b.material]
    ax = np.array([math.cos(b.ang), math.sin(b.ang)])
    nx = np.array([-ax[1], ax[0]])
    hl, hw = b.length / 2, b.width / 2
    a, bb = b.c - ax * hl - nx * hw, b.c + ax * hl - nx * hw
    cc, d = b.c + ax * hl + nx * hw, b.c - ax * hl + nx * hw
    r = hl - hw if b.hip else hl
    r1, r2 = b.c - ax * r, b.c + ax * r
    planes = [(np.array([a, bb, r2, r1]), -nx), (np.array([d, cc, r2, r1]), nx)]
    if b.hip:
        planes += [(np.array([a, d, r1]), -ax), (np.array([bb, cc, r2]), ax)]
    wear = rng.uniform(0.9, 1.04)
    grain = rng.random((96, 160)).astype(np.float32)
    for poly, hdir in planes:
        n3 = np.array([hdir[0] * math.sin(b.pitch), hdir[1] * math.sin(b.pitch), math.cos(b.pitch)])
        light = 0.26 + 0.95 * max(0.0, float(n3 @ LIGHT))

        def colour(xx, yy, hdir=hdir, light=light):
            s = (xx - b.c[0]) * hdir[0] + (yy - b.c[1]) * hdir[1]  # down the slope
            t = (yy - b.c[1]) * hdir[0] - (xx - b.c[0]) * hdir[1]  # along the eave
            tex = 1 + contrast * np.sin(2 * np.pi * (t if ribs else s) / period)
            g = cv2.resize(grain, (xx.shape[1], xx.shape[0]), interpolation=cv2.INTER_LINEAR)
            return np.array(rgb, np.float32) * (light * wear * tex * (0.96 + 0.08 * g))[..., None]

        _paint(img, poly, colour)
    ridge = np.array(rgb, np.float32) * 1.12
    for p, q in [(r1, r2)] + ([(a, r1), (d, r1), (bb, r2), (cc, r2)] if b.hip else []):
        _line(img, p, q, 1.3, ridge, 0.7)
    eave = np.array(rgb, np.float32) * 0.55
    for p, q in ((a, bb), (bb, cc), (cc, d), (d, a)):
        _line(img, p, q, 1.2, eave, 0.6)
    if b.solar:  # dark panel array on the sunnier roof plane
        sunny = nx if float(nx @ LIGHT[:2]) > 0 else -nx
        c = b.c + sunny * hw * 0.5
        _paint(img, _rect(c[0], c[1], b.length * 0.45, hw * 0.6, b.ang), (44, 52, 66))
        for k in (-1, 0, 1):
            p = c + ax * b.length * 0.15 * k
            _paint(img, _rect(p[0], p[1], 0.8, hw * 0.6, b.ang), (92, 100, 112), 0.6)
    if b.chimney:
        c = b.c + ax * hl * rng.uniform(-0.6, 0.6) + nx * hw * 0.25
        s = c + _cast(1.2)
        _paint(img, _rect(s[0], s[1], 4, 4, b.ang), (40, 36, 34), 0.5)
        _paint(img, _rect(c[0], c[1], 4, 4, b.ang), (112, 72, 58))


def _crown_bank(rng: np.random.Generator, n: int = 40) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tree crowns as height fields of a few overlapping domes (lumpy, like real foliage), lit by
    the sun through their surface normals and broken up by leaf clumps. Every fifth is a conifer."""
    greens = ((66, 88, 48), (78, 98, 54), (58, 80, 46), (88, 104, 58), (70, 84, 52), (84, 92, 50), (60, 76, 56))
    bank = []
    for i in range(n):
        conifer = i % 5 == 0
        r = rng.uniform(7, 11) if conifer else rng.uniform(12, 26)
        size = int(math.ceil(r * 2.7)) | 1
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) - size / 2
        height = np.zeros((size, size), np.float32)
        if conifer:
            th = np.arctan2(yy, xx)
            ph = rng.uniform(0, 6.3, 2)
            rr = r * (1 + 0.06 * np.sin(9 * th + ph[0]) + 0.04 * np.sin(5 * th + ph[1]))
            height = np.sqrt(np.clip(1 - (xx**2 + yy**2) / rr**2, 0, 1)) * r * 1.4
        else:
            for _ in range(int(rng.integers(3, 7))):
                lr = r * rng.uniform(0.5, 0.75)
                ox, oy = rng.normal(0, r * 0.28, 2)
                dome = np.sqrt(np.clip(1 - ((xx - ox) ** 2 + (yy - oy) ** 2) / lr**2, 0, 1)) * lr
                height = np.maximum(height, dome)
        alpha = np.clip(height / 1.5, 0, 1).astype(np.float32)
        gy, gx = np.gradient(cv2.GaussianBlur(height, (0, 0), 0.8))
        nrm = np.dstack([-gx, -gy, np.ones_like(height)])
        nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
        lit = np.clip(nrm @ LIGHT, 0, 1)
        clumps = np.zeros((size, size), np.float32)
        for sigma, weight in ((max(1.0, r / 6), 0.6), (max(0.7, r / 14), 0.4)):  # leaf clusters, two scales
            c = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), sigma)
            clumps += weight * (c - c.min()) / (np.ptp(c) + 1e-6)
        fine = rng.random((size, size)).astype(np.float32)
        base = np.array((44, 62, 46) if conifer else greens[int(rng.integers(len(greens)))], np.float32)
        base *= rng.uniform(0.9, 1.1)
        shade = (0.3 + 0.6 * lit + 0.55 * (clumps - 0.5) * (0.5 + lit)) * (0.9 + 0.2 * fine)
        shade = np.clip(shade, 0.12, 1.5)
        bank.append(((base * shade[..., None]).astype(np.float32), alpha))
    return bank


def _terrain(w: int, h: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Procedural countryside at GSD_M m/px. Returns (BGR image, road centre-lines)."""
    rng = np.random.default_rng(seed * 7919 + 17)
    img, hedges = _fields(h, w, rng)
    img *= (0.9 + 0.2 * _fbm(h, w, rng, 260))[..., None]  # soil and moisture
    img *= (0.94 + 0.12 * _fbm(h, w, rng, 6, octaves=3))[..., None]  # grain
    forest = _smoothstep(0.62, 0.66, _fbm(h, w, rng, 380))
    pm = _fbm(h, w, rng, 260)
    pond = _smoothstep(0.866, 0.876, pm) * (1 - forest)
    reeds = _smoothstep(0.85, 0.86, pm) * (1 - pond) * (1 - forest)
    img += (np.array((84, 96, 56), np.float32) - img) * (reeds * 0.8)[..., None]
    img += (np.array((42, 50, 34), np.float32) - img) * forest[..., None]  # woodland floor
    water = np.array((48, 62, 62), np.float32) * (0.94 + 0.1 * _fbm(h, w, rng, 40, octaves=3))[..., None]
    img += (water - img) * pond[..., None]

    # a farm track with two wheel ruts, then two asphalt roads: verge, worn lanes, painted lines
    track = _bezier((-40, rng.uniform(0.1, 0.9) * h), (rng.uniform(0, w), rng.uniform(0, h)),
                    (rng.uniform(0, w), rng.uniform(0, h)), (w + 40, rng.uniform(0.1, 0.9) * h))
    ruts = _stroke((h, w), [_offset(track, -4.0), _offset(track, 4.0)], 3)
    img += (np.array((136, 120, 96), np.float32) - img) * (ruts * 0.8)[..., None]
    roads = [
        _bezier((-60, rng.uniform(0.25, 0.75) * h), (w * 0.33, rng.uniform(0, h)),
                (w * 0.66, rng.uniform(0, h)), (w + 60, rng.uniform(0.25, 0.75) * h)),
        _bezier((rng.uniform(0.25, 0.75) * w, -60), (rng.uniform(0, w), h * 0.33),
                (rng.uniform(0, w), h * 0.66), (rng.uniform(0.25, 0.75) * w, h + 60)),
    ]
    verge, asphalt = _stroke((h, w), roads, 47), _stroke((h, w), roads, 35)
    img += (np.array((150, 144, 124), np.float32) - img) * (verge * 0.8)[..., None]
    tar = np.array((86, 88, 90), np.float32) * (0.9 + 0.2 * _fbm(h, w, rng, 3, octaves=2))[..., None]
    img += (tar - img) * asphalt[..., None]
    wheels = [_offset(p, lane + wheel) for p in roads for lane in (-8.7, 8.7) for wheel in (-3.6, 3.6)]
    edges = [_offset(p, side) for p in roads for side in (-15.5, 15.5)]
    dashes = []
    for pts in roads:  # centre line: 3 m dashes, 9 m gaps
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        dash = (arc % 60.0) < 15.0
        dashes += [pts[r] for r in np.split(np.arange(len(pts)), np.flatnonzero(np.diff(dash.astype(int))) + 1)
                   if dash[r[0]] and len(r) > 1]
    paint = np.maximum(_stroke((h, w), edges, 1) * 0.7, _stroke((h, w), dashes, 1) * 0.85)
    patches = _smoothstep(0.62, 0.7, _fbm(h, w, rng, 40, octaves=3))  # repaired, darker tar
    worn = cv2.GaussianBlur(_stroke((h, w), wheels, 3), (0, 0), 1.2)
    img *= (1 - (0.07 * worn + 0.08 * patches) * asphalt)[..., None]
    img += (np.array((222, 222, 214), np.float32) - img) * paint[..., None]

    # buildings with yards and driveways; trees wherever there is no road, roof or water
    buildings = _plan_buildings(roads, forest, w, h, rng)
    for b in buildings:
        _yard(img, b, rng)
    occupied = cv2.dilate((verge > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8))
    for b in buildings:
        foot = _rect(b.c[0], b.c[1], b.length + 24, b.width + 24, b.ang)
        cv2.fillConvexPoly(occupied, np.round(foot).astype(np.int32), 1)
    clear = (occupied > 0) | (pond > 0.2)
    gy, gx = np.mgrid[8:h - 8:16, 8:w - 8:16].astype(np.float32)
    grid = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-6, 6, (gx.size, 2))
    iy, ix = grid[:, 1].astype(int).clip(0, h - 1), grid[:, 0].astype(int).clip(0, w - 1)
    keep = (rng.random(len(grid)) < forest[iy, ix] * 0.95) | (rng.random(len(grid)) < 0.0035)
    trees = list(grid[keep])
    for line in hedges:
        trees += [p + rng.normal(0, 2.5, 2) for p in line if rng.random() < 0.8]
    for b in buildings:
        if b.house:  # a couple of garden trees
            for _ in range(int(rng.integers(0, 3))):
                t = rng.uniform(0, 2 * math.pi)
                trees.append(b.c + np.array([math.cos(t), math.sin(t)]) * (math.hypot(b.length, b.width) / 2 + 22))
    trees = [p for p in trees if 0 <= p[0] < w and 0 <= p[1] < h and not clear[int(p[1]), int(p[0])]]

    bank = _crown_bank(rng)
    shadow = np.zeros((h, w), np.float32)
    crown_c = np.zeros((h, w, 3), np.float32)
    crown_a = np.zeros((h, w), np.float32)
    for p in trees:
        col, alpha = bank[int(rng.integers(len(bank)))]
        s = alpha.shape[0]
        for (ox, oy), layer in ((tuple(_cast(rng.uniform(9, 18))), None), ((0.0, 0.0), col)):
            x0, y0 = int(round(p[0] + ox)) - s // 2, int(round(p[1] + oy)) - s // 2
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, w), min(y0 + s, h)
            if xb <= xa or yb <= ya:
                continue
            a = alpha[ya - y0:yb - y0, xa - x0:xb - x0]
            if layer is None:
                np.maximum(shadow[ya:yb, xa:xb], a, out=shadow[ya:yb, xa:xb])
            else:
                c = layer[ya - y0:yb - y0, xa - x0:xb - x0]
                crown_c[ya:yb, xa:xb] += (c - crown_c[ya:yb, xa:xb]) * a[..., None]
                crown_a[ya:yb, xa:xb] += a * (1 - crown_a[ya:yb, xa:xb])
    bshadow = np.zeros((h, w), np.uint8)
    for b in buildings:
        foot = _rect(b.c[0], b.c[1], b.length, b.width, b.ang)
        hull = cv2.convexHull(np.round(np.concatenate([foot, foot + _cast(b.height_m)]) * ONE).astype(np.int32))
        cv2.fillConvexPoly(bshadow, hull, 255, AA, SHIFT)
    shadow = np.maximum(cv2.GaussianBlur(shadow, (0, 0), 2.2), cv2.GaussianBlur(bshadow, (0, 0), 1.0) / 255.0)
    img *= (1 - 0.5 * shadow)[..., None]
    crown_c *= (1 - 0.3 * shadow)[..., None]  # crowns down-sun of taller neighbours sit in their shade
    img += (crown_c - img) * crown_a[..., None]
    for b in buildings:
        _roof(img, b, rng)

    # EO camera look: muted colour and a little atmospheric haze at altitude
    img = (img - 110.0) * 1.1 + 110.0
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * 0.8
    img = img * 0.93 + np.array((176, 182, 188), np.float32) * 0.07
    return np.ascontiguousarray(np.clip(img[..., ::-1], 0, 255).astype(np.uint8)), roads


_AERIAL = Path(__file__).resolve().parent / "data" / "vallei"


@lru_cache(maxsize=1)
def _aerial() -> tuple[np.ndarray, dict] | None:
    """The bundled aerial photo and its metadata (scripts/build_aerial.py), if installed."""
    try:
        meta = json.loads(_AERIAL.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    img = cv2.imread(str(_AERIAL.with_suffix(".webp")), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img.setflags(write=False)
    return img, meta


def _ground(w: int, h: int, seed: int, view: str, background: str
            ) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Ground for a world of w x h px: a window of the aerial photo centred on the scene's
    view, or the procedural countryside. Returns (BGR image, road centre-lines, the image
    direction shadows fall in)."""
    aerial = _aerial() if background == "photo" else None
    if aerial is not None:
        img, meta = aerial
        ph, pw = img.shape[:2]
        if w <= pw and h <= ph:
            cx, cy = meta.get("views", {}).get(view, (pw // 2, ph // 2))
            x0, y0 = int(np.clip(cx - w // 2, 0, pw - w)), int(np.clip(cy - h // 2, 0, ph - h))
            return img[y0:y0 + h, x0:x0 + w], (), np.asarray(meta["sun_dir"], float)
    terrain, roads = _terrain_cached(w, h, seed)
    return terrain, roads, SUN_DIR


@lru_cache(maxsize=4)
def _terrain_cached(w: int, h: int, seed: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Terrain is a pure function of size and seed; scenes built again (tests, reconnects) reuse it."""
    img, roads = _terrain(w, h, seed)
    img.setflags(write=False)
    return img, tuple(roads)


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


def _draw_board(img: np.ndarray, m: _Mover, dx: float, dy: float, sun: np.ndarray = SUN) -> None:
    _shade(img, _board_corners(m, dx, dy) + sun, 0.42)
    for scale, rgb in ((1.0, (214, 214, 206)), (0.84, (40, 40, 42)), (0.56, (214, 214, 206)),
                       (0.30, (164, 52, 46))):
        _fill(img, _board_corners(m, dx, dy, scale), _rgb(*rgb))


def _draw_car(img: np.ndarray, m: _Mover, dx: float, dy: float) -> None:
    """Top-down car at real proportions (about 4.6 x 1.9 m): body with rounded corners, dark
    cabin sides, windscreen and rear window, a roof catching the sun, a soft shadow."""
    cx, cy, length, ang = m.x - dx, m.y - dy, m.size, m.angle
    width = length * 0.42
    ca, sa = math.cos(ang), math.sin(ang)

    def part(a0: float, a1: float, frac: float, colour, chamfer: float = 0.0) -> None:
        mid = (a0 + a1) / 2 * length  # a0..a1: section of the length, +0.5 is the front
        _fill(img, _rect(cx + ca * mid, cy + sa * mid, (a1 - a0) * length, width * frac, ang, chamfer), colour)

    _shade(img, _rect(cx, cy, length, width, ang, width * 0.3) + _cast(1.5), 0.45)
    body = m.color
    glass = _rgb(46, 54, 64)
    part(-0.5, 0.5, 1.0, body, width * 0.32)
    part(-0.30, 0.20, 0.9, tuple(int(v * 0.55) for v in body))  # side windows and pillars
    part(0.06, 0.20, 0.82, glass)  # windscreen
    part(-0.30, -0.20, 0.76, glass)  # rear window
    part(-0.20, 0.06, 0.74, tuple(min(255, int(v * 1.06 + 6)) for v in body))  # roof


CAR_COLORS = [(226, 226, 222), (232, 232, 230), (172, 174, 178), (128, 130, 134), (52, 54, 58), (30, 31, 34),
              (46, 60, 92), (104, 36, 38), (186, 178, 158)]


class _World:
    """Static terrain plus the moving objects that live on it."""

    def __init__(self, width: int, height: int, seed: int, n_targets: int, n_cars: int,
                 bounds: tuple[float, float, float, float], occlusion: bool,
                 background: str = "photo", view: str = "fixed"):
        self.rng = np.random.default_rng(seed)
        self.w, self.h = width, height
        self.terrain, roads, sun_dir = _ground(width, height, seed, view, background)
        self.roads = list(roads)
        self.sun = sun_dir * 3.0  # shadow offset of the boards, along the scene's own sun
        n_cars = n_cars if self.roads else 0  # the photo has its own (parked) vehicles
        self.road_len = [np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
                         for r in self.roads]
        self.bounds = bounds
        self.target_bounds = bounds  # where targets roam; PanningScene narrows it to the gimbal's reach
        self.separate_targets = False  # ground targets cannot drive through each other
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
                x=0, y=0, size=r.uniform(21, 24), kind="car", road=road,
                s=r.uniform(0, self.road_len[road][-1]),
                speed=r.choice([-1, 1]) * r.uniform(1.6, 3.0),
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
            lane = 8.7 if m.speed > 0 else -8.7  # centre of the right-hand lane
            m.x, m.y = x - hy / norm * lane, y + hx / norm * lane
            m.angle = math.atan2(hy, hx) + (0.0 if m.speed > 0 else math.pi)

    def update(self) -> None:
        bx0, by0, bx1, by1 = self.target_bounds
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
        if self.separate_targets:
            self._separate()
        self._place_cars()
        self.t += 1

    def _separate(self) -> None:
        """Elastic bounce between targets that are about to overlap."""
        targets = [m for m in self.movers if m.kind == "target"]
        for i, a in enumerate(targets):
            for b in targets[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy) or 1e-6
                if dist >= 0.8 * (a.size + b.size):
                    continue
                nx, ny = dx / dist, dy / dist
                va, vb = a.vx * nx + a.vy * ny, b.vx * nx + b.vy * ny
                if va > vb:  # closing: swap the velocity components along the line between them
                    a.vx, a.vy = a.vx + (vb - va) * nx, a.vy + (vb - va) * ny
                    b.vx, b.vy = b.vx + (va - vb) * nx, b.vy + (va - vb) * ny

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
            if m.kind == "car":
                _draw_car(view, m, ix, iy)
            else:
                _draw_board(view, m, ix, iy, self.sun)
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
        background: str = "photo",
    ):
        self.w, self.h = width, height
        self.pad = 60
        self.world = _World(width + 2 * self.pad, height + 2 * self.pad, seed, n_targets, n_distractors,
                            (self.pad, self.pad, self.pad + width, self.pad + height), occlusion,
                            background, "fixed")
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
                 n_targets: int = 2, n_distractors: int = 4, background: str = "photo"):
        self.w, self.h = width, height
        self.mw, self.mh = int(width * margin), int(height * margin)
        ww, wh = width + 2 * self.mw, height + 2 * self.mh
        self.world = _World(ww, wh, seed, n_targets, n_distractors, (8, 8, ww - 8, wh - 8), occlusion,
                            background, "panning")
        self.rng = np.random.default_rng(seed + 1)
        self.optics = _Optics(width, height, noise, self.rng)
        self.ppd = width / hfov_deg
        self._ref: tuple[float, float] | None = None
        # Targets roam only where the gimbal can bring them to the image centre. The world is
        # rendered just `margin` beyond the view, so past that the simulated gimbal sits on an
        # end stop and the target slides out of the lock zone - a limit of the simulation, not
        # of the tracker, which the demo should not show as a lost lock.
        self.world.target_bounds = (width / 2, height / 2, width / 2 + 2 * self.mw, height / 2 + 2 * self.mh)
        self.world.separate_targets = True
        for m in self.world.movers:  # start every target inside the initial field of view
            if m.kind == "target":
                m.x = float(self.rng.uniform(self.mw + 0.25 * width, self.mw + 0.75 * width))
                m.y = float(self.rng.uniform(self.mh + 0.25 * height, self.mh + 0.75 * height))
                # road-vehicle pace (about 20-45 km/h at 0.2 m/px and 30 fps): the area the gimbal
                # can reach is under 200 m across, so faster targets would only bounce off its edges
                m.vx *= 0.6
                m.vy *= 0.6

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
