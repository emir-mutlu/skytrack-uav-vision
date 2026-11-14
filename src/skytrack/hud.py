"""Operator HUD overlay rendered with OpenCV."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .lock import LockState, LockStatus, TargetLock
from .types import Track

GREEN = (80, 220, 100)
AMBER = (0, 190, 255)
RED = (60, 60, 240)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
CYAN = (230, 200, 60)

STATE_COLOR = {
    LockState.SEARCHING: GREY,
    LockState.ACQUIRING: AMBER,
    LockState.LOCKED: RED,
    LockState.LOST: AMBER,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, txt, org, color=WHITE, scale=0.5, thick=1):
    # Same-thickness offset shadow: a thicker stroke would widen glyph advance in OpenCV.
    cv2.putText(img, txt, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def _text_right(img, txt, right, y, color=WHITE, scale=0.5, thick=1):
    (tw, _), _ = cv2.getTextSize(txt, FONT, scale, thick)
    _text(img, txt, (right - tw, y), color, scale, thick)


def _corners(img, bbox, color, length=0.25, thick=2):
    x1, y1, x2, y2 = map(int, bbox)
    lx, ly = int((x2 - x1) * length), int((y2 - y1) * length)
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * lx, py), color, thick, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * ly), color, thick, cv2.LINE_AA)


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
) -> np.ndarray:
    img = frame
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Lock zone
    zx1, zy1, zx2, zy2 = map(int, locker.zone_px(img.shape))
    zone_color = STATE_COLOR[lock.state]
    _corners(img, (zx1, zy1, zx2, zy2), zone_color, length=0.08, thick=2)

    # Boresight crosshair
    g = 14
    cv2.line(img, (cx - 3 * g, cy), (cx - g, cy), WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (cx + g, cy), (cx + 3 * g, cy), WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 3 * g), (cx, cy - g), WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + g), (cx, cy + 3 * g), WHITE, 1, cv2.LINE_AA)

    # Tracks
    for tr in tracks:
        is_target = tr.track_id == lock.track_id
        is_cand = tr.track_id == lock.candidate_id and not is_target
        color = STATE_COLOR[lock.state] if is_target else (CYAN if is_cand else GREEN)
        x1, y1, x2, y2 = map(int, tr.bbox)
        if is_target:
            _corners(img, tr.bbox, color, thick=3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        if draw_trails and len(tr.history) > 1:
            pts = np.array(tr.history, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        tag = f"#{tr.track_id} {tr.label} {tr.score:.2f}"
        if is_cand:
            tag += "  [candidate]"
        _text(img, tag, (x1, max(12, y1 - 6)), color, 0.45)

    # Centring error vector + acquisition progress
    if lock.bbox is not None and lock.track_id is not None:
        tx = int((lock.bbox[0] + lock.bbox[2]) / 2)
        ty = int((lock.bbox[1] + lock.bbox[3]) / 2)
        cv2.arrowedLine(img, (cx, cy), (tx, ty), STATE_COLOR[lock.state], 1, cv2.LINE_AA, tipLength=0.05)
        if lock.state == LockState.ACQUIRING:
            x1, y1, x2, y2 = map(int, lock.bbox)
            bw = x2 - x1
            cv2.rectangle(img, (x1, y2 + 4), (x2, y2 + 9), GREY, 1)
            cv2.rectangle(img, (x1, y2 + 4), (x1 + int(bw * lock.progress), y2 + 9), AMBER, -1)

    # Top status bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    state_txt = lock.state.value
    if lock.state == LockState.LOCKED:
        state_txt += f"  #{lock.track_id}  {lock.lock_duration:4.1f}s"
    _text(img, state_txt, (10, 21), STATE_COLOR[lock.state], 0.6, 2)
    err = f"err x{lock.error[0]:+.2f} y{lock.error[1]:+.2f}" if lock.error else "err --"
    info = f"{detector_name.upper()} | tracks {len(tracks)} | {err} | {fps:5.1f} FPS | {latency_ms:5.1f} ms"
    _text_right(img, info, w - 10, 21, WHITE, 0.45)

    # Footer
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _text(img, stamp, (10, h - 10), GREY, 0.42)
    help_txt = "[click/TAB] designate  [R] release  [SPACE] pause  [Q] quit"
    _text_right(img, help_txt, w - 10, h - 10, GREY, 0.42)
    if paused:
        _text(img, "PAUSED", (cx - 40, 60), AMBER, 0.8, 2)
    return img
