"""Gimbal follow controller: turns the target's image-plane error into pitch/yaw rate
commands that keep the designated target centred in the camera.

The normalised pixel error is first converted to an angular error through the lens
model, so the gains are expressed in physical units ((deg/s) per deg) and stay valid
when the camera, resolution or zoom changes.
"""

from __future__ import annotations

import math

from .config import GimbalConfig


class GimbalController:
    def __init__(self, cfg: GimbalConfig, hfov_deg: float, aspect: float):
        self.cfg = cfg
        self.tan_h = math.tan(math.radians(hfov_deg) / 2.0)
        self.tan_v = self.tan_h * aspect
        self._prev: tuple[float, float] | None = None
        self._integ = [0.0, 0.0]
        self._t_prev: float | None = None
        self._t_sent: float | None = None
        self.active = False  # True while a target is being followed

    def reset(self) -> None:
        self._prev, self._integ, self._t_prev = None, [0.0, 0.0], None

    def angular_error(self, err: tuple[float, float]) -> tuple[float, float]:
        """(yaw_err_deg, pitch_err_deg); pitch_err is positive when the target is above centre."""
        ex, ey = err
        return math.degrees(math.atan(ex * self.tan_h)), -math.degrees(math.atan(ey * self.tan_v))

    def update(self, err: tuple[float, float] | None, t: float) -> tuple[float, float] | None:
        """Return (pitch_rate_dps, yaw_rate_dps) to send now, or None if nothing to send.

        When the target disappears a single zero-rate command is emitted so the gimbal
        stops instead of continuing its last motion."""
        c = self.cfg
        if err is None:
            if self.active:
                self.active = False
                self.reset()
                self._t_sent = t
                return (0.0, 0.0)
            return None
        if self._t_sent is not None and t - self._t_sent < 1.0 / c.rate_hz:
            return None

        ex, ey = (0.0 if abs(e) < c.deadband else e for e in err)
        yaw_e, pitch_e = self.angular_error((ex, ey))
        dt = (t - self._t_prev) if self._t_prev is not None else 0.0
        out = []
        for i, e in enumerate((pitch_e, yaw_e)):
            d = 0.0
            if self._prev is not None and dt > 1e-4:
                d = (e - self._prev[i]) / dt
            if c.ki:
                self._integ[i] += e * dt
                limit = c.max_rate_dps / c.ki
                self._integ[i] = max(-limit, min(limit, self._integ[i]))
            u = c.kp * e + c.ki * self._integ[i] + c.kd * d
            out.append(max(-c.max_rate_dps, min(c.max_rate_dps, u)))
        self._prev, self._t_prev, self._t_sent = (pitch_e, yaw_e), t, t
        self.active = True
        return out[0], out[1]
