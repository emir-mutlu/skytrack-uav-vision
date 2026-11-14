"""Target-lock state machine.

A track is *lock-eligible* when it is confirmed, of an allowed class, large enough and
fully inside the lock zone. Holding an eligible target for `acquire_seconds` declares
LOCKED; the lock survives short dropouts (`lost_grace_seconds`) via the tracker's
Kalman prediction.

Human-in-the-loop by design: with `require_operator_confirm=True` the system only
*proposes* a candidate; acquisition starts after the operator designates a track.
The module outputs camera/gimbal centring errors and lock events only - it contains
no actuation or weapon-release logic.

    SEARCHING --eligible target--> ACQUIRING --held acquire_seconds--> LOCKED
        ^                              |                                  |
        +------- grace expired --------+---------- LOST <-- grace expired-+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import LockConfig
from .types import BBox, Track


class LockState(str, Enum):
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    LOST = "LOST"


@dataclass
class LockEvent:
    kind: str  # lock_acquired | lock_lost | designated | released
    t: float
    track_id: int | None
    duration: float | None = None


@dataclass
class LockStatus:
    state: LockState = LockState.SEARCHING
    track_id: int | None = None
    candidate_id: int | None = None
    bbox: BBox | None = None
    progress: float = 0.0  # 0..1 acquisition progress
    error: tuple[float, float] | None = None  # normalised (-1..1) offset from frame centre
    lock_duration: float = 0.0
    events: list[LockEvent] = field(default_factory=list)


class TargetLock:
    def __init__(self, cfg: LockConfig | None = None):
        self.cfg = cfg or LockConfig()
        self.state = LockState.SEARCHING
        self.target_id: int | None = None
        self.designated_id: int | None = None
        self._acquire_start: float | None = None
        self._lock_start: float | None = None
        self._last_seen: float | None = None
        self.history: list[LockEvent] = []

    # ---------------------------------------------------------------- operator API
    def designate(self, track_id: int, t: float) -> LockEvent:
        self.designated_id = track_id
        if self.target_id != track_id:
            self._reset_target()
        return self._emit(LockEvent("designated", t, track_id))

    def release(self, t: float) -> LockEvent:
        ev = LockEvent("released", t, self.target_id or self.designated_id)
        self.designated_id = None
        self._reset_target()
        return self._emit(ev)

    # ---------------------------------------------------------------- geometry
    def zone_px(self, shape: tuple[int, ...]) -> BBox:
        h, w = shape[:2]
        x1, y1, x2, y2 = self.cfg.zone
        return (x1 * w, y1 * h, x2 * w, y2 * h)

    def is_eligible(self, tr: Track, shape: tuple[int, ...]) -> bool:
        if not tr.confirmed:
            return False
        if self.cfg.target_labels and tr.label not in self.cfg.target_labels:
            return False
        h, w = shape[:2]
        bw, bh = tr.size
        if max(bw / w, bh / h) < self.cfg.min_size_frac:
            return False
        zx1, zy1, zx2, zy2 = self.zone_px(shape)
        x1, y1, x2, y2 = tr.bbox
        return x1 >= zx1 and y1 >= zy1 and x2 <= zx2 and y2 <= zy2

    @staticmethod
    def centre_error(tr: Track, shape: tuple[int, ...]) -> tuple[float, float]:
        h, w = shape[:2]
        cx, cy = tr.center
        return ((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2))

    # ---------------------------------------------------------------- main update
    def update(self, tracks: list[Track], shape: tuple[int, ...], t: float) -> LockStatus:
        status = LockStatus()
        if not self.cfg.enabled:
            return status
        by_id = {tr.track_id: tr for tr in tracks}
        eligible = [tr for tr in tracks if self.is_eligible(tr, shape)]

        # Best candidate = eligible track closest to the image centre.
        def dist(tr: Track) -> float:
            ex, ey = self.centre_error(tr, shape)
            return ex * ex + ey * ey

        candidate = min(eligible, key=dist) if eligible else None
        status.candidate_id = candidate.track_id if candidate else None

        # Which track are we working on?
        if self.designated_id is not None:
            wanted = self.designated_id
        elif not self.cfg.require_operator_confirm:
            if self.target_id in by_id and by_id[self.target_id] in eligible:
                wanted = self.target_id
            else:
                wanted = candidate.track_id if candidate else self.target_id
        else:
            wanted = None

        if wanted != self.target_id and self.state not in (LockState.LOCKED,):
            self._reset_target()
            self.target_id = wanted
        elif self.target_id is None:
            self.target_id = wanted

        target = by_id.get(self.target_id) if self.target_id is not None else None
        events_before = len(self.history)

        if target is not None and self.is_eligible(target, shape):
            self._last_seen = t
            if self.state in (LockState.SEARCHING, LockState.LOST):
                self.state = LockState.ACQUIRING
                self._acquire_start = t
            if self.state == LockState.ACQUIRING:
                assert self._acquire_start is not None
                if t - self._acquire_start >= self.cfg.acquire_seconds:
                    self.state = LockState.LOCKED
                    self._lock_start = t
                    self._emit(LockEvent("lock_acquired", t, self.target_id))
        elif self.state in (LockState.ACQUIRING, LockState.LOCKED):
            if self._last_seen is None or t - self._last_seen > self.cfg.lost_grace_seconds:
                if self.state == LockState.LOCKED:
                    dur = t - (self._lock_start or t)
                    self._emit(LockEvent("lock_lost", t, self.target_id, round(dur, 3)))
                    self.state = LockState.LOST
                else:
                    self.state = LockState.SEARCHING
                self._acquire_start = None
                self._lock_start = None
        elif self.state == LockState.LOST:
            if self._last_seen is None or t - self._last_seen > 3 * self.cfg.lost_grace_seconds:
                self.state = LockState.SEARCHING
                if self.designated_id is not None and self.designated_id not in by_id:
                    self.designated_id = None  # track is gone for good
                self.target_id = None

        status.state = self.state
        status.track_id = self.target_id
        if target is not None:
            status.bbox = target.bbox
            status.error = self.centre_error(target, shape)
        if self.state == LockState.ACQUIRING and self._acquire_start is not None:
            status.progress = min(1.0, (t - self._acquire_start) / max(self.cfg.acquire_seconds, 1e-6))
        elif self.state == LockState.LOCKED:
            status.progress = 1.0
            status.lock_duration = t - (self._lock_start or t)
        status.events = self.history[events_before:]
        return status

    # ---------------------------------------------------------------- helpers
    def _reset_target(self) -> None:
        self.state = LockState.SEARCHING
        self.target_id = None
        self._acquire_start = None
        self._lock_start = None
        self._last_seen = None

    def _emit(self, ev: LockEvent) -> LockEvent:
        self.history.append(ev)
        return ev
