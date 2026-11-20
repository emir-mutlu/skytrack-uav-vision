"""Payload video symbology (operator HUD).

Monochrome symbology in the style of EO/IR gimbal payloads: boresight reticle, track
gates, sensor azimuth tape, gimbal elevation scale and corner data blocks. Geometry is
drawn with anti-aliased OpenCV primitives; text is rendered from cached TrueType glyphs
with a black outline, so it stays crisp and readable over any terrain. Every size scales with frame height, so the
same layout works from 480p ground-station previews to 4K recordings.

Fonts are picked from the system (Bahnschrift / Cascadia Mono on Windows, DejaVu on
Linux, Helvetica / Menlo on macOS) and can be overridden with the environment
variables SKYTRACK_FONT_SANS, SKYTRACK_FONT_BOLD and SKYTRACK_FONT_MONO.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geo import mgrs
from .lock import LockState, LockStatus, TargetLock
from .types import BBox, Track

if TYPE_CHECKING:
    from .geo import GeoPoint
    from .vehicle import VehicleState

# Palette (BGR - text is drawn straight onto the BGR buffer, so Pillow gets BGR too)
WHITE = (242, 242, 242)
GREY = (175, 175, 175)
DIM = (125, 125, 125)
OUTLINE = (16, 16, 16)
PANEL = (24, 19, 15)
TEAL = (160, 215, 70)
CYAN = (240, 205, 80)
AMBER = (40, 182, 255)
RED = (72, 62, 240)

STATE_COLOR = {
    LockState.SEARCHING: GREY,
    LockState.ACQUIRING: AMBER,
    LockState.LOCKED: RED,
    LockState.LOST: AMBER,
}

PANEL_ALPHA = 0.62
AA = cv2.LINE_AA

# ----------------------------------------------------------------------------- fonts
# B612 Mono - the open cockpit-display face Airbus developed with ENAC (SIL OFL 1.1) - ships
# with the package, so the symbology looks the same on every machine; system faces are fallbacks.
_B612 = Path(__file__).resolve().parent / "console" / "static"
_FONT_CANDIDATES = {
    "sans": (str(_B612 / "B612Mono-Regular.ttf"), "bahnschrift.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"),
    "bold": (str(_B612 / "B612Mono-Bold.ttf"), "bahnschrift.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"),
    "mono": (str(_B612 / "B612Mono-Regular.ttf"), "CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf"),
}


@lru_cache(maxsize=64)
def get_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    env = os.environ.get(f"SKYTRACK_FONT_{kind.upper()}")
    for name in ((env,) if env else ()) + _FONT_CANDIDATES[kind]:
        try:
            f = ImageFont.truetype(name, size)
        except OSError:
            continue
        if kind == "bold" and name.lower().startswith("bahnschrift"):
            try:  # variable font: pick the bold instance
                f.set_variation_by_name("Bold")
            except Exception:  # noqa: BLE001 - FreeType built without variation support
                pass
        return f
    return ImageFont.load_default(size=size)


@dataclass
class _Text:
    xy: tuple[float, float]
    text: str
    color: tuple[int, int, int]
    size: int
    kind: str = "sans"
    anchor: str = "la"
    stroke: int = 0


# ----------------------------------------------------------------------------- canvas
class _Canvas:
    """Collects translucent panels and text, then composites each layer in one pass."""

    def __init__(self, img: np.ndarray):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.s = self.h / 720.0
        self._overlay: np.ndarray | None = None
        self._texts: list[_Text] = []

    def px(self, v: float) -> int:
        return max(1, int(round(v * self.s)))

    def fs(self, v: float) -> int:
        return max(10, int(round(v * self.s)))

    # -- layers
    @property
    def overlay(self) -> np.ndarray:
        if self._overlay is None:
            self._overlay = self.img.copy()
        return self._overlay

    def panel(self, rect: tuple[int, int, int, int], radius: int | None = None,
              color: tuple[int, int, int] = PANEL) -> None:
        _rounded_rect(self.overlay, *rect, self.px(8) if radius is None else radius, color)

    def flush_overlay(self, alpha: float = PANEL_ALPHA) -> None:
        if self._overlay is not None:
            cv2.addWeighted(self._overlay, alpha, self.img, 1 - alpha, 0, dst=self.img)
            self._overlay = None

    def text(self, xy, text, color=WHITE, size=15, kind="sans", anchor="la", stroke=0) -> None:
        self._texts.append(_Text(xy, text, color, self.fs(size), kind, anchor, stroke))

    def text_width(self, text: str, size: float, kind: str = "sans") -> int:
        return _sprite(text, kind, self.fs(size), 0)[2]

    def flush_text(self) -> None:
        for t in self._texts:
            fill, stroke_mask, tw = _sprite(t.text, t.kind, t.size, t.stroke)
            asc, desc = get_font(t.kind, t.size).getmetrics()
            ax = {"l": 0.0, "m": tw / 2, "r": float(tw)}[t.anchor[0]]
            ay = {"a": 0.0, "m": (asc + desc) / 2, "d": float(asc + desc), "s": float(asc)}[t.anchor[1]]
            x0 = int(round(t.xy[0] - ax)) - _PAD - t.stroke
            y0 = int(round(t.xy[1] - ay)) - _PAD - t.stroke
            if stroke_mask is not None:
                _blit(self.img, stroke_mask, OUTLINE, x0, y0)
            _blit(self.img, fill, t.color, x0, y0)
        self._texts.clear()


# ----------------------------------------------------------------------------- text sprites
# Rasterising text with FreeType costs ~0.2 ms per label, i.e. several ms per frame.
# Labels are therefore rendered once into cached alpha masks; monospaced strings
# (all the changing numbers) are assembled from cached glyphs, so a new value costs
# a few NumPy ops instead of a FreeType call.
_PAD = 2


def _render(text: str, kind: str, size: int, stroke: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    font = get_font(kind, size)
    asc, desc = font.getmetrics()
    tw = int(math.ceil(font.getlength(text)))
    p = _PAD + stroke
    shape = (tw + 2 * p + 1, asc + desc + 2 * p)
    fill = Image.new("L", shape, 0)
    ImageDraw.Draw(fill).text((p, p), text, fill=255, font=font, anchor="la")
    stroke_mask = None
    if stroke:
        sm = Image.new("L", shape, 0)
        ImageDraw.Draw(sm).text((p, p), text, fill=255, font=font, anchor="la",
                                stroke_width=stroke, stroke_fill=255)
        stroke_mask = np.asarray(sm, np.float32) / 255.0
    return np.asarray(fill, np.float32) / 255.0, stroke_mask, tw


@lru_cache(maxsize=1024)
def _glyph(ch: str, kind: str, size: int, stroke: int):
    return _render(ch, kind, size, stroke)


@lru_cache(maxsize=32)
def _mono_advance(kind: str, size: int) -> float | None:
    font = get_font(kind, size)
    adv = font.getlength("0")
    return adv if abs(font.getlength("i") - adv) < 0.01 and abs(font.getlength("W") - adv) < 0.01 else None


@lru_cache(maxsize=4096)
def _sprite(text: str, kind: str, size: int, stroke: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    adv = _mono_advance(kind, size) if kind == "mono" and len(text) > 1 else None
    if adv is None:
        return _render(text, kind, size, stroke)
    p = _PAD + stroke
    gh = _glyph("0", kind, size, stroke)[0].shape[0]
    tw = int(math.ceil(adv * len(text)))
    fill = np.zeros((gh, tw + 2 * p + 1 + int(math.ceil(adv))), np.float32)
    smask = np.zeros_like(fill) if stroke else None
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        g_fill, g_stroke, _ = _glyph(ch, kind, size, stroke)
        x = int(round(i * adv))
        gw = min(g_fill.shape[1], fill.shape[1] - x)
        np.maximum(fill[:, x:x + gw], g_fill[:, :gw], out=fill[:, x:x + gw])
        if smask is not None and g_stroke is not None:
            np.maximum(smask[:, x:x + gw], g_stroke[:, :gw], out=smask[:, x:x + gw])
    return fill, smask, tw


def _blit(img: np.ndarray, mask: np.ndarray, color, x0: int, y0: int) -> None:
    h, w = img.shape[:2]
    mh, mw = mask.shape
    x1, y1, x2, y2 = max(0, x0), max(0, y0), min(w, x0 + mw), min(h, y0 + mh)
    if x2 <= x1 or y2 <= y1:
        return
    a = mask[y1 - y0:y2 - y0, x1 - x0:x2 - x0, None]
    roi = img[y1:y2, x1:x2].astype(np.float32)
    roi += (np.asarray(color, np.float32) - roi) * a
    img[y1:y2, x1:x2] = (roi + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- primitives
def _rounded_rect(img, x1, y1, x2, y2, r, color) -> None:
    r = int(max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, AA)


def _line(img, p1, p2, color, t, outline=True) -> None:
    if outline:
        cv2.line(img, p1, p2, OUTLINE, t + 2, AA)
    cv2.line(img, p1, p2, color, t, AA)


def _brackets(img, bbox: BBox, color, frac: float, t: int, outline: bool = True) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    lx = max(4, int((x2 - x1) * frac))
    ly = max(4, int((y2 - y1) * frac))
    segs = [
        ((x1, y1), (x1 + lx, y1)), ((x1, y1), (x1, y1 + ly)),
        ((x2, y1), (x2 - lx, y1)), ((x2, y1), (x2, y1 + ly)),
        ((x1, y2), (x1 + lx, y2)), ((x1, y2), (x1, y2 - ly)),
        ((x2, y2), (x2 - lx, y2)), ((x2, y2), (x2, y2 - ly)),
    ]
    if outline:
        for a, b in segs:
            cv2.line(img, a, b, OUTLINE, t + 2, AA)
    for a, b in segs:
        cv2.line(img, a, b, color, t, AA)


def _dashed(img, p1, p2, color, t, dash: int, gap: int) -> None:
    (x1, y1), (x2, y2) = p1, p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    d = 0.0
    while d < length:
        e = min(d + dash, length)
        cv2.line(img, (int(x1 + ux * d), int(y1 + uy * d)), (int(x1 + ux * e), int(y1 + uy * e)),
                 color, t, AA)
        d += dash + gap


# ----------------------------------------------------------------------------- symbology
# Monochrome white symbology with a black outline, as burned into EO/IR payload video:
# readable over sky, terrain and snow alike, and nothing competes with the scene for colour.
SYM = WHITE
SYM_DIM = (196, 196, 196)
WARN = AMBER


def _dms(value: float, pos: str, neg: str, deg_digits: int) -> str:
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60.0) * 3600.0
    return f"{d:0{deg_digits}d}°{m:02d}'{s:04.1f}\"{pos if value >= 0 else neg}"


def _grid(lat: float, lon: float) -> str:
    """MGRS (10 m) as military video overlays show positions; DMS where MGRS is undefined."""
    try:
        return mgrs(lat, lon)
    except ValueError:
        return latlon_dms(lat, lon)


def latlon_dms(lat: float, lon: float) -> str:
    return f"{_dms(lat, 'N', 'S', 2)} {_dms(lon, 'E', 'W', 3)}"


def _tri(img, pts, color) -> None:
    p = np.array(pts, np.int32)
    cv2.fillConvexPoly(img, p, OUTLINE, AA)
    cv2.polylines(img, [p], True, OUTLINE, 3, AA)
    cv2.fillConvexPoly(img, p, color, AA)


def _readout(cv: _Canvas, img, x: float, y: float, text: str, size: float = 13) -> None:
    """Boxed numeric readout (black box, white frame), as on a heading or elevation tape."""
    px = cv.px
    tw = cv.text_width(text, size, "mono")
    hw, hh = tw / 2 + px(6), px(size * 0.75)
    cv2.rectangle(img, (int(x - hw), int(y - hh)), (int(x + hw), int(y + hh)), OUTLINE, -1)
    cv2.rectangle(img, (int(x - hw), int(y - hh)), (int(x + hw), int(y + hh)), SYM, 1, AA)
    cv.text((x, y), text, SYM, size, "mono", "mm")


def _heading_tape(cv: _Canvas, img, cx: int, top: int, width: int, heading: float) -> None:
    px, span = cv.px, 30.0
    ppd = (width / 2) / span
    base = top + px(16)
    for d in range(int(math.floor(heading - span)), int(math.ceil(heading + span)) + 1):
        if d % 5:
            continue
        x = cx + (d - heading) * ppd
        major = d % 10 == 0
        _line(img, (int(x), base), (int(x), base + (px(9) if major else px(5))), SYM, 1)
        if major:
            label = {0: "N", 90: "E", 180: "S", 270: "W"}.get(d % 360, f"{d % 360:03d}")
            cv.text((x, base - px(3)), label, SYM, 11, "mono", "md", stroke=1)
    _tri(img, [(cx, base + px(11)), (cx - px(5), base + px(19)), (cx + px(5), base + px(19))], SYM)
    _readout(cv, img, cx, base + px(32), f"{heading % 360:05.1f}")


def _elevation_scale(cv: _Canvas, img, x: int, cy: int, height: int, pitch: float) -> None:
    px = cv.px
    top = cy - height / 2

    def ypos(p: float) -> int:
        return int(top + (min(max(-p, 0.0), 90.0) / 90.0) * height)

    _line(img, (x, ypos(0)), (x, ypos(-90)), SYM, 1)
    for p in range(0, -91, -10):
        major = p % 30 == 0
        _line(img, (x - (px(9) if major else px(5)), ypos(p)), (x, ypos(p)), SYM, 1)
        if major:
            cv.text((x - px(13), ypos(p)), f"{p}", SYM, 11, "mono", "rm", stroke=1)
    y = ypos(pitch)
    _tri(img, [(x + px(3), y), (x + px(11), y - px(5)), (x + px(11), y + px(5))], SYM)
    cv.text((x + px(15), y), f"{pitch:+.0f}", SYM, 12, "mono", "lm", stroke=1)


def _gate(cv: _Canvas, img, t: Track, lock: LockStatus, is_target: bool, auto: bool) -> None:
    px = cv.px
    x1, y1, x2, y2 = map(int, t.bbox)
    if not is_target:
        _brackets(img, t.bbox, SYM_DIM, 0.18, 1)
        mark = f"{t.track_id}" + ("?" if t.track_id == lock.candidate_id else "")
        cv.text((x1, y1 - px(4)), mark, SYM_DIM, 11, "mono", "ld", stroke=1)
        return
    pad = px(4)
    gate = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    if lock.state == LockState.LOST:
        corners = [(gate[0], gate[1]), (gate[2], gate[1]), (gate[2], gate[3]), (gate[0], gate[3])]
        for i in range(4):
            _dashed(img, corners[i], corners[(i + 1) % 4], WARN, 1, px(5), px(5))
        cv.text((gate[0], gate[1] - px(4)), f"{t.track_id} LOST", WARN, 12, "mono", "ld", stroke=1)
        return
    _brackets(img, gate, SYM, 0.3, px(2))
    if lock.state == LockState.LOCKED:
        cv2.rectangle(img, (gate[0], gate[1]), (gate[2], gate[3]), SYM, 1, AA)
        tag = "LOCK"
        tw = cv.text_width(tag, 12, "mono") + px(8)
        tx, ty = gate[0], gate[1] - px(19)
        cv2.rectangle(img, (tx, ty), (tx + tw, ty + px(15)), SYM, -1)
        cv.text((tx + px(4), ty + px(8)), tag, OUTLINE, 12, "mono", "lm")
        cv.text((tx + tw + px(5), ty + px(8)), f"{t.track_id}", SYM, 12, "mono", "lm", stroke=1)
    else:  # acquiring: the gate is set, lock not yet declared
        cv.text((gate[0], gate[1] - px(4)), f"{t.track_id} ACQ {lock.progress * 100:3.0f}%", SYM, 12, "mono", "ld",
                stroke=1)


# ----------------------------------------------------------------------------- HUD
def draw_hud(
    frame: np.ndarray,
    tracks: list[Track],
    lock: LockStatus,
    locker: TargetLock,
    fps: float,
    latency_ms: float,
    detector_name: str,
    draw_trails: bool = True,
    paused: bool = False,
    profile: str = "",
    frame_idx: int | None = None,
    vehicle: VehicleState | None = None,
    target_geo: GeoPoint | None = None,
    gimbal_follow: bool = False,
    link_timeout: float = 3.0,
    panels: bool = True,
) -> np.ndarray:
    """Draw payload-style video symbology.

    Always: boresight reticle, acquisition window, track gates, and - when vehicle data is
    available - the sensor azimuth tape and gimbal elevation scale. With `panels=True` the
    corner data blocks (time, platform, sensor, target) are burned in as well; the web
    console turns them off because it shows that data in its own panes."""
    cv = _Canvas(frame)
    img, h, w, px = frame, cv.h, cv.w, cv.px
    cx, cy = w // 2, h // 2
    pad = px(14)
    lock_on = locker.cfg.enabled
    link_ok = vehicle is not None and vehicle.connected(link_timeout)

    # acquisition window (lock zone): small grey corner ticks
    if lock_on:
        zx1, zy1, zx2, zy2 = map(int, locker.zone_px(frame.shape))
        _brackets(img, (zx1, zy1, zx2, zy2), SYM_DIM, 0.04, 1)

    # boresight reticle
    g, length = px(9), px(28)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _line(img, (cx + dx * g, cy + dy * g), (cx + dx * (g + length), cy + dy * (g + length)), SYM, 1)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # end ticks
        ex, ey = cx + dx * (g + length), cy + dy * (g + length)
        _line(img, (ex - dy * px(4), ey - dx * px(4)), (ex + dy * px(4), ey + dx * px(4)), SYM, 1)
    cv2.circle(img, (cx, cy), 1, SYM, -1, AA)

    # track gates
    auto = not locker.cfg.require_operator_confirm
    target = None
    for t in tracks:
        is_target = t.track_id == lock.track_id and lock_on
        if is_target:
            target = t
        if draw_trails and is_target and len(t.history) > 2:
            for p in t.history[:-1:3]:
                cv2.circle(img, p, 1, SYM_DIM, -1, AA)
        _gate(cv, img, t, lock, is_target, auto)

    # sensor azimuth tape and gimbal elevation scale
    if link_ok and vehicle.heading_deg is not None:
        az = (vehicle.heading_deg + (vehicle.gimbal_yaw_deg or 0.0)) % 360.0
        _heading_tape(cv, img, cx, pad, int(w * 0.36), az)
    if link_ok and vehicle.gimbal_pitch_deg is not None:
        _elevation_scale(cv, img, w - px(70), cy, int(h * 0.42), vehicle.gimbal_pitch_deg)

    # tracker mode, bottom centre
    if not lock_on:
        mode = "TRACK ONLY"
    else:
        mode = {LockState.SEARCHING: "SEARCH", LockState.ACQUIRING: "ACQUIRE", LockState.LOCKED: "TRACK",
                LockState.LOST: "COAST"}[lock.state]
    status = mode + ("  AUTO" if auto and lock_on else "  MAN" if lock_on else "")
    if paused:
        status = "FREEZE  " + status
    cv.text((cx, h - pad), status, WARN if lock.state == LockState.LOST else SYM, 15, "mono", "md", stroke=1)

    if panels:
        _data_blocks(cv, w, h, pad, lock, target, fps, latency_ms, detector_name, frame_idx, vehicle,
                     target_geo, gimbal_follow, link_ok, profile)
    cv.flush_text()
    return img


def _data_blocks(cv: _Canvas, w: int, h: int, pad: int, lock: LockStatus, target: Track | None, fps: float,
                 latency_ms: float, detector: str, frame_idx: int | None, vehicle, geo, follow: bool,
                 link_ok: bool, profile: str) -> None:
    px, size = cv.px, 13
    lh = px(18)

    def block(lines: list[str], x: float, y: float, anchor: str, upward: bool = False) -> None:
        rows = list(reversed(lines)) if upward else lines
        for i, line in enumerate(rows):
            cv.text((x, y - i * lh if upward else y + i * lh), line, SYM, size, "mono", anchor, stroke=1)

    left = [f"EO  {detector.upper()}  {(profile or 'default').upper()}"]
    if vehicle is not None:
        if link_ok:
            left += [f"{vehicle.mode or '---'}  {'ARMED' if vehicle.armed else 'DISARMED'}"]
            if vehicle.alt_rel is not None:
                left += [f"ALT {vehicle.alt_rel:4.0f}M AGL  {vehicle.alt_amsl or 0:4.0f}M MSL"]
            if vehicle.groundspeed is not None:
                hdg = f"{vehicle.heading_deg:03.0f}" if vehicle.heading_deg is not None else "---"
                left += [f"GS {vehicle.groundspeed:4.1f}M/S  HDG {hdg}"]
        else:
            left += ["NO LINK"]
    block(left, pad, pad, "la")

    stamp = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
    right = [stamp, f"FRM {frame_idx or 0:06d}  {fps:4.1f}FPS", f"PROC {latency_ms:4.1f}MS"]
    block(right, w - pad, pad, "ra")

    bl: list[str] = []
    if vehicle is not None and link_ok and vehicle.lat is not None:
        bl.append(f"AC  {_grid(vehicle.lat, vehicle.lon)}")
        if vehicle.gimbal_pitch_deg is not None:
            bl.append(f"GMB AZ{vehicle.gimbal_yaw_deg or 0:+06.1f} EL{vehicle.gimbal_pitch_deg:+05.1f}"
                      + ("  FOLLOW" if follow else "  HOLD"))
    if bl:
        block(bl, pad, h - pad - px(4), "ld", upward=True)

    br: list[str] = []
    if target is not None:
        state = {LockState.LOCKED: f"LOCK {lock.lock_duration:4.1f}S", LockState.ACQUIRING: "ACQ",
                 LockState.LOST: "LOST"}.get(lock.state, "")
        br.append(f"TGT {target.track_id}  {state}")
        if geo is not None:
            br.append(_grid(geo.lat, geo.lon))
            br.append(f"SLR {geo.slant_range_m:4.0f}M  GR {geo.ground_range_m:4.0f}M  {geo.method.upper()}")
        elif lock.error is not None:
            br.append(f"ERR X{lock.error[0]:+.2f} Y{lock.error[1]:+.2f}")
    if br:
        block(br, w - pad, h - pad - px(4), "rd", upward=True)
