"""Operator HUD overlay.

Geometry (brackets, reticle, arcs, translucent panels) is drawn with anti-aliased
OpenCV primitives; all text is rendered in a single Pillow pass with TrueType fonts,
so labels stay crisp at any resolution. Every size scales with frame height, so the
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
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
_FONT_CANDIDATES = {
    "sans": ("bahnschrift.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"),
    "bold": ("bahnschrift.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"),
    "mono": ("CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "Courier New.ttf"),
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


def _track_color(t: Track, lock: LockStatus, state_col) -> tuple[int, int, int]:
    if t.track_id == lock.track_id:
        return state_col
    return CYAN if t.track_id == lock.candidate_id else TEAL


def _vehicle_rows(vs: VehicleState, geo: GeoPoint | None, follow: bool,
                  timeout: float) -> list[tuple[str, str, tuple[int, int, int]]]:
    if not vs.connected(timeout):
        return [("LINK", "NO HEARTBEAT", AMBER)]
    rows = [("LINK", f"{vs.mode or '?'} · {'ARMED' if vs.armed else 'DISARMED'}",
             TEAL if vs.armed else GREY)]
    if vs.has_position:
        rows.append(("POS", f"{vs.lat:.6f} {vs.lon:.6f}", WHITE))
        hdg = f"{vs.heading_deg:03.0f}°" if vs.heading_deg is not None else "---"
        rows.append(("ALT", f"{vs.alt_rel:5.1f} m   HDG {hdg}", WHITE))
        if vs.groundspeed is not None:
            rows.append(("GS", f"{vs.groundspeed:4.1f} m/s", WHITE))
    if vs.gimbal_pitch_deg is not None:
        rows.append(("GMB", f"P{vs.gimbal_pitch_deg:+.1f}°  Y{vs.gimbal_yaw_deg or 0.0:+.1f}°", WHITE))
    if follow:
        rows.append(("CAM", "FOLLOWING TARGET", CYAN))
    if geo is not None:
        rows.append(("TGT", f"{geo.lat:.6f} {geo.lon:.6f}", RED))
        rows.append(("RNG", f"{geo.ground_range_m:.0f} m ground", RED))
    return rows


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
) -> np.ndarray:
    cv = _Canvas(frame)
    img, h, w, px = frame, cv.h, cv.w, cv.px
    cx, cy = w // 2, h // 2
    pad = px(10)
    lock_on = locker.cfg.enabled
    state_col = STATE_COLOR[lock.state] if lock_on else GREY

    # ------------------------------------------------------------ layer 1: translucent panels
    panel_w, panel_h = px(330), px(56)
    tl = (pad, pad, pad + panel_w, pad + panel_h)
    tr = (w - pad - panel_w, pad, w - pad, pad + panel_h)
    box = px(136)
    er = (w - pad - box, h - pad - box - px(46), w - pad, h - pad)
    bl = (pad, h - pad - px(30), pad + px(290), h - pad)
    help_txt = "CLICK / TAB  designate     R  release     SPACE  pause     Q  quit"
    hw = cv.text_width(help_txt, 11) + px(28)
    hb = (cx - hw // 2, h - pad - px(26), cx + hw // 2, h - pad)
    for rect in (tl, tr, er, bl, hb):
        cv.panel(rect)
    vrows: list[tuple[str, str, tuple[int, int, int]]] = []
    if vehicle is not None:
        vrows = _vehicle_rows(vehicle, target_geo, gimbal_follow, link_timeout)
        row_h = px(19)
        vp = (pad, tl[3] + px(8), pad + px(232), tl[3] + px(8) + px(12) + row_h * len(vrows))
        cv.panel(vp)
    if lock_on:
        zx1, zy1, zx2, zy2 = map(int, locker.zone_px(frame.shape))
        zl_w = cv.text_width("LOCK ZONE", 10) + px(14)
        zone_tag = (zx1 + px(6), zy2 - px(24), zx1 + px(6) + zl_w, zy2 - px(6))
        cv.panel(zone_tag, radius=px(4))

    tag_h = px(20)
    tags: list[tuple[Track, tuple[int, int, int, int], str, tuple[int, int, int]]] = []
    for t in tracks:
        is_target = t.track_id == lock.track_id
        is_cand = t.track_id == lock.candidate_id and not is_target
        color = _track_color(t, lock, state_col)
        label = f"#{t.track_id}  {t.label.upper()}  {t.score:.2f}"
        if is_cand:
            label += "  CANDIDATE"
        x1, y1, _, y2 = map(int, t.bbox)
        tw = cv.text_width(label, 12, "mono") + px(14)
        ty = y1 - px(8) - tag_h
        if ty < tl[3] + px(4):  # no room above: put the tag under the box
            ty = y2 + px(8)
        tx = int(np.clip(x1, 0, w - tw))
        rect = (tx, ty, tx + tw, ty + tag_h)
        cv.panel(rect, radius=px(4))
        tags.append((t, rect, label, color))

    if draw_trails:  # trails share the translucent layer -> soft, semi-transparent paths
        for t in tracks:
            hist = t.history
            n = len(hist)
            for i in range(1, n):
                thick = px(2) if i > n // 2 else 1
                cv2.line(cv.overlay, hist[i - 1], hist[i], _track_color(t, lock, state_col), thick, AA)
    cv.flush_overlay()

    # ------------------------------------------------------------ layer 2: geometry
    if lock_on:
        zone_frac = min(zx2 - zx1, zy2 - zy1) * 0.07 / max(zx2 - zx1, 1)
        _brackets(img, (zx1, zy1, zx2, zy2), state_col, zone_frac, px(2))
        cv.text(((zone_tag[0] + zone_tag[2]) / 2, (zone_tag[1] + zone_tag[3]) / 2), "LOCK ZONE", GREY, 10,
                "sans", "mm")

    # boresight reticle
    r1, r2 = px(9), px(30)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _line(img, (cx + dx * r1, cy + dy * r1), (cx + dx * r2, cy + dy * r2), WHITE, px(1.5))
    cv2.circle(img, (cx, cy), px(2), WHITE, -1, AA)

    target_center = None
    for t, _, _, color in tags:
        is_target = t.track_id == lock.track_id
        x1, y1, x2, y2 = map(int, t.bbox)
        tcx, tcy = (x1 + x2) // 2, (y1 + y2) // 2
        if is_target:
            target_center = (tcx, tcy)
            _brackets(img, t.bbox, color, 0.3, px(2.5))
            if lock.state == LockState.LOCKED:
                pulse = px(6) + int(px(3) * abs(math.sin(lock.lock_duration * 5.0)))
                cv2.rectangle(img, (x1 - pulse, y1 - pulse), (x2 + pulse, y2 + pulse), color, 1, AA)
                for (mx, my) in ((tcx, y1 - pulse), (tcx, y2 + pulse), (x1 - pulse, tcy), (x2 + pulse, tcy)):
                    cv2.circle(img, (mx, my), px(2.5), color, -1, AA)
            elif lock.state == LockState.ACQUIRING:
                r = int(max(x2 - x1, y2 - y1) * 0.5 + px(14))
                cv2.circle(img, (tcx, tcy), r, DIM, 1, AA)
                cv2.ellipse(img, (tcx, tcy), (r, r), -90, 0, 360 * lock.progress, AMBER, px(3), AA)
        else:
            _brackets(img, t.bbox, color, 0.22, px(1.5))
        # predicted motion over the next ~0.5 s
        vx, vy = t.velocity
        if vx * vx + vy * vy > 0.09:
            end = (int(tcx + vx * 15), int(tcy + vy * 15))
            cv2.line(img, (tcx, tcy), end, color, 1, AA)
            cv2.circle(img, end, px(2.5), color, 1, AA)

    if target_center is not None:
        _dashed(img, (cx, cy), target_center, state_col, 1, px(8), px(6))

    # boresight-error indicator (bottom right)
    ix1, iy1 = er[0] + px(10), er[1] + px(24)
    inner = box - px(20)
    ix2, iy2 = ix1 + inner, iy1 + inner
    icx, icy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
    cv2.rectangle(img, (ix1, iy1), (ix2, iy2), DIM, 1, AA)
    cv2.line(img, (icx, iy1), (icx, iy2), DIM, 1, AA)
    cv2.line(img, (ix1, icy), (ix2, icy), DIM, 1, AA)
    cv2.circle(img, (icx, icy), inner // 4, DIM, 1, AA)
    if lock.error is not None:
        ex, ey = (float(np.clip(e, -1, 1)) for e in lock.error)
        dot = (int(icx + ex * inner / 2), int(icy + ey * inner / 2))
        cv2.line(img, (icx, icy), dot, state_col, 1, AA)
        cv2.circle(img, dot, px(4), state_col, -1, AA)
        err_txt = f"X{ex:+.2f}  Y{ey:+.2f}"
    else:
        err_txt = "X --    Y --"
    cv.text(((er[0] + er[2]) / 2, er[1] + px(13)), "BORESIGHT ERROR", GREY, 10, "sans", "mm")
    cv.text(((er[0] + er[2]) / 2, er[3] - px(13)), err_txt, WHITE, 11, "mono", "mm")

    # ------------------------------------------------------------ solid pills + panel text
    # status panel (top left)
    if not lock_on:
        pill_txt, sub = "TRACK ONLY", f"{len(tracks)} ACTIVE TRACKS"
    else:
        pill_txt = lock.state.value
        if lock.state == LockState.LOCKED:
            sub = f"TGT #{lock.track_id}   LOCK {lock.lock_duration:5.1f} s"
        elif lock.state == LockState.ACQUIRING:
            sub = f"TGT #{lock.track_id}   ACQ {lock.progress * 100:3.0f} %"
        elif lock.state == LockState.LOST:
            sub = f"TGT #{lock.track_id}   REACQUIRING"
        elif lock.candidate_id is not None:
            sub = f"CANDIDATE #{lock.candidate_id}"
        else:
            sub = "NO TARGET"
    pill_w = cv.text_width(pill_txt, 14, "bold") + px(20)
    pill = (tl[0] + px(10), tl[1] + px(8), tl[0] + px(10) + pill_w, tl[1] + px(31))
    _rounded_rect(img, *pill, px(5), state_col)
    cv.text(((pill[0] + pill[2]) / 2, (pill[1] + pill[3]) / 2), pill_txt, PANEL, 14, "bold", "mm")
    cv.text((pill[2] + px(12), (pill[1] + pill[3]) / 2), sub, WHITE, 14, "mono", "lm")
    mode = "OPERATOR DESIGNATION" if locker.cfg.require_operator_confirm else "AUTO DESIGNATION"
    prof = (profile or "default").upper()
    cv.text((tl[0] + px(11), tl[3] - px(12)), f"PROFILE {prof}   ·   {mode}", GREY, 11, "sans", "lm")

    # telemetry panel (top right)
    cells = (("FPS", f"{fps:.1f}"), ("LATENCY", f"{latency_ms:.1f} ms"),
             ("TRACKS", str(len(tracks))), ("DETECTOR", detector_name.upper()))
    cw = (tr[2] - tr[0]) / len(cells)
    for i, (k, v) in enumerate(cells):
        x = tr[0] + cw * i + cw / 2
        cv.text((x, tr[1] + px(17)), k, GREY, 10, "sans", "mm")
        cv.text((x, tr[1] + px(38)), v, WHITE, 14, "mono", "mm")
        if i:
            xx = int(tr[0] + cw * i)
            cv2.line(img, (xx, tr[1] + px(12)), (xx, tr[3] - px(12)), (70, 70, 70), 1, AA)

    # vehicle panel (left, under the status panel)
    for i, (k, v, col) in enumerate(vrows):
        y = vp[1] + px(6) + row_h * i + row_h / 2
        cv.text((vp[0] + px(10), y), k, GREY, 10, "sans", "lm")
        cv.text((vp[0] + px(46), y), v, col, 12, "mono", "lm")

    # footer (bottom left): UTC clock + frame counter
    stamp = time.strftime("%Y-%m-%d  %H:%M:%S", time.gmtime())
    foot = f"{stamp} UTC" + (f"   F{frame_idx:06d}" if frame_idx is not None else "")
    cv.text((bl[0] + px(12), (bl[1] + bl[3]) / 2), foot, GREY, 12, "mono", "lm")

    # track tags
    for _, rect, label, color in tags:
        cv.text((rect[0] + px(7), (rect[1] + rect[3]) / 2), label, color, 12, "mono", "lm")

    cv.text((cx, (hb[1] + hb[3]) / 2), help_txt, GREY, 11, "sans", "mm")

    if paused:
        pw = cv.text_width("PAUSED", 16, "bold") + px(28)
        pr = (cx - pw // 2, tl[3] + px(12), cx + pw // 2, tl[3] + px(42))
        _rounded_rect(img, *pr, px(6), AMBER)
        cv.text((cx, (pr[1] + pr[3]) / 2), "PAUSED", PANEL, 16, "bold", "mm")

    cv.flush_text()
    return img
