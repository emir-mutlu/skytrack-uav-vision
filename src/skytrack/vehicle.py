"""MAVLink link to an ArduPilot / PX4 flight controller.

Runs a background reader that keeps the latest vehicle state (position, attitude,
gimbal attitude, mode) and exposes a small command surface for the camera gimbal.
SkyTrack identifies itself as an onboard computer (MAV_COMP_ID_ONBOARD_COMPUTER) and
only ever sends:

* HEARTBEAT (1 Hz)
* MAV_CMD_SET_MESSAGE_INTERVAL - to request the telemetry streams it needs
* MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW - gimbal pitch/yaw *rates* for camera follow

It never commands vehicle motion, flight modes or arming.

Works with ArduPilot SITL, MAVProxy / mavlink-router outputs, telemetry radios and
direct serial links (USB or companion-computer UART).
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from .config import MavlinkConfig

STREAM_MSG_IDS = (33, 30, 285, 158)  # GLOBAL_POSITION_INT, ATTITUDE, GIMBAL_DEVICE_ATTITUDE_STATUS, MOUNT_STATUS


def _mavutil():
    os.environ.setdefault("MAVLINK20", "1")  # gimbal v2 messages need MAVLink 2
    try:
        from pymavlink import mavutil
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("MAVLink support requires: pip install -e .[mavlink]") from exc
    return mavutil


@dataclass
class VehicleState:
    last_heartbeat: float | None = None  # time.monotonic()
    armed: bool = False
    mode: str = ""
    lat: float | None = None
    lon: float | None = None
    alt_amsl: float | None = None
    alt_rel: float | None = None  # metres above home (used as AGL on flat terrain)
    heading_deg: float | None = None
    groundspeed: float | None = None
    roll: float = 0.0  # rad
    pitch: float = 0.0
    yaw: float = 0.0
    gimbal_pitch_deg: float | None = None
    gimbal_yaw_deg: float | None = None  # relative to the vehicle nose
    gimbal_source: str = ""

    def connected(self, timeout: float = 3.0) -> bool:
        return self.last_heartbeat is not None and time.monotonic() - self.last_heartbeat < timeout

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None and self.alt_rel is not None

    def as_dict(self) -> dict[str, Any]:
        r = lambda v, n=6: None if v is None else round(v, n)  # noqa: E731
        return {
            "connected": self.connected(), "armed": self.armed, "mode": self.mode,
            "lat": r(self.lat, 7), "lon": r(self.lon, 7), "alt_rel": r(self.alt_rel, 2),
            "heading": r(self.heading_deg, 1), "groundspeed": r(self.groundspeed, 2),
            "roll": r(math.degrees(self.roll), 2), "pitch": r(math.degrees(self.pitch), 2),
            "yaw": r(math.degrees(self.yaw), 2),
            "gimbal_pitch": r(self.gimbal_pitch_deg, 2), "gimbal_yaw": r(self.gimbal_yaw_deg, 2),
        }


def quat_to_euler(q) -> tuple[float, float, float]:
    """MAVLink quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians (ZYX)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class MavlinkLink:
    def __init__(self, cfg: MavlinkConfig):
        self.cfg = cfg
        self._state = VehicleState()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._streams_requested = False
        self._have_gimbal_v2 = False
        self.conn = None
        self.messages_rx = 0
        self.commands_tx = 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> MavlinkLink:
        mavutil = _mavutil()
        self._mav = mavutil.mavlink
        self.conn = mavutil.mavlink_connection(
            self.cfg.connection, baud=self.cfg.baud, source_system=self.cfg.source_system,
            source_component=self.cfg.source_component, autoreconnect=True,
        )
        self._mode_string = mavutil.mode_string_v10
        self._thread = threading.Thread(target=self._loop, name="mavlink-rx", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.conn is not None:
            self.conn.close()

    def wait_connected(self, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            st = self.state
            if st.connected(self.cfg.heartbeat_timeout) and st.has_position:
                return True
            time.sleep(0.05)
        return False

    @property
    def state(self) -> VehicleState:
        with self._lock:
            return replace(self._state)

    # ------------------------------------------------------------------ commands
    def send_gimbal_rates(self, pitch_dps: float, yaw_dps: float) -> None:
        nan = float("nan")
        self._command(self._mav.MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW,
                      nan, nan, float(pitch_dps), float(yaw_dps), 0, 0, 0)

    def _command(self, cmd: int, *params: float) -> None:
        p = list(params) + [0.0] * (7 - len(params))
        with self._send_lock:
            self.conn.mav.command_long_send(self.cfg.target_system, self.cfg.target_component, cmd, 0, *p)
        self.commands_tx += 1

    def _request_streams(self) -> None:
        interval_us = int(1e6 / max(self.cfg.stream_hz, 0.1))
        for msg_id in STREAM_MSG_IDS:
            self._command(self._mav.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, interval_us)
        self._streams_requested = True

    # ------------------------------------------------------------------ rx loop
    def _loop(self) -> None:
        mav = self._mav
        last_hb = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_hb >= 1.0:
                with self._send_lock:
                    self.conn.mav.heartbeat_send(mav.MAV_TYPE_ONBOARD_CONTROLLER, mav.MAV_AUTOPILOT_INVALID,
                                                 0, 0, mav.MAV_STATE_ACTIVE)
                last_hb = now
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except OSError:
                time.sleep(0.2)
                continue
            if msg is None or msg.get_type() == "BAD_DATA":
                continue
            if msg.get_srcSystem() != self.cfg.target_system:
                continue
            self.messages_rx += 1
            self._handle(msg)

    def _handle(self, msg) -> None:
        mav = self._mav
        kind = msg.get_type()
        with self._lock:
            s = self._state
            if kind == "HEARTBEAT":
                if msg.autopilot == mav.MAV_AUTOPILOT_INVALID:
                    return  # GCS, gimbal or another companion computer
                s.last_heartbeat = time.monotonic()
                s.armed = bool(msg.base_mode & mav.MAV_MODE_FLAG_SAFETY_ARMED)
                try:
                    s.mode = self._mode_string(msg)
                except Exception:  # noqa: BLE001 - unknown autopilot/mode mapping
                    s.mode = str(msg.custom_mode)
                request = not self._streams_requested
            elif kind == "GLOBAL_POSITION_INT":
                s.lat, s.lon = msg.lat / 1e7, msg.lon / 1e7
                s.alt_amsl, s.alt_rel = msg.alt / 1000.0, msg.relative_alt / 1000.0
                s.heading_deg = None if msg.hdg == 65535 else msg.hdg / 100.0
                s.groundspeed = math.hypot(msg.vx, msg.vy) / 100.0
                return
            elif kind == "ATTITUDE":
                s.roll, s.pitch, s.yaw = msg.roll, msg.pitch, msg.yaw
                return
            elif kind == "GIMBAL_DEVICE_ATTITUDE_STATUS":
                _, gp, gy = quat_to_euler(msg.q)
                f = msg.flags
                earth = bool(f & mav.GIMBAL_DEVICE_FLAGS_YAW_IN_EARTH_FRAME) or (
                    not f & mav.GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME and f & mav.GIMBAL_DEVICE_FLAGS_YAW_LOCK
                )
                yaw_deg = math.degrees(gy)
                if earth:
                    yaw_deg -= math.degrees(s.yaw)
                s.gimbal_pitch_deg, s.gimbal_yaw_deg = math.degrees(gp), wrap180(yaw_deg)
                s.gimbal_source = "gimbal_v2"
                self._have_gimbal_v2 = True
                return
            elif kind == "MOUNT_STATUS" and not self._have_gimbal_v2:
                s.gimbal_pitch_deg = msg.pointing_a / 100.0
                s.gimbal_yaw_deg = wrap180(msg.pointing_c / 100.0)
                s.gimbal_source = "mount_status"
                return
            else:
                return
        if request:
            self._request_streams()
