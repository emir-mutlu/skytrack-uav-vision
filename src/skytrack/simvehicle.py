"""Minimal MAVLink vehicle for demos, CI and development without hardware.

It is *not* a flight simulator - use ArduPilot SITL for that - but it speaks the same
protocol subset SkyTrack relies on: an armed copter flying a circle at constant altitude,
streaming HEARTBEAT, GLOBAL_POSITION_INT, ATTITUDE and GIMBAL_DEVICE_ATTITUDE_STATUS,
and a gimbal that integrates MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW rate commands.

    skytrack sim-vehicle      # serves MAVLink on tcp 127.0.0.1:5760, like ArduPilot SITL
    skytrack sim-vehicle --connection udpout:127.0.0.1:14550   # like a MAVProxy UDP output
"""

from __future__ import annotations

import math
import threading
import time

from .vehicle import _mavutil, open_connection, wrap180

SITL_HOME = (-35.363261, 149.165230, 584.0)  # ArduPilot's default SITL home (lat, lon, amsl)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


class SimVehicle:
    def __init__(
        self,
        connection: str = "tcpin:127.0.0.1:5760",
        home: tuple[float, float, float] = SITL_HOME,
        altitude: float = 100.0,
        radius: float = 150.0,
        speed: float = 10.0,
        rate_hz: float = 20.0,
        gimbal_pitch_deg: float = -45.0,
        system_id: int = 1,
        pitch_limits: tuple[float, float] = (-90.0, 30.0),
        yaw_limits: tuple[float, float] | None = None,
    ):
        self.connection = connection
        self.home, self.altitude, self.radius, self.speed = home, altitude, radius, speed
        self.rate_hz = rate_hz
        self.system_id = system_id
        self.gimbal_pitch = gimbal_pitch_deg
        self.pitch_limits, self.yaw_limits = pitch_limits, yaw_limits
        self.gimbal_yaw = 0.0
        self._rates = (0.0, 0.0)
        self._rate_t = 0.0
        self.gimbal_commands = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.conn = None

    def start(self) -> SimVehicle:
        mavutil = _mavutil()
        self._mav = mavutil.mavlink
        self.conn = open_connection(self.connection, source_system=self.system_id, source_component=1)
        self._thread = threading.Thread(target=self._loop, name="sim-vehicle", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.conn is not None:
            self.conn.close()

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        mav = self._mav
        t0 = last = time.monotonic()
        last_hb = last_status = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            dt, t = now - last, now - t0
            last = now
            self._drain()
            if now - self._rate_t > 1.0:  # gimbal managers stop on command timeout
                self._rates = (0.0, 0.0)
            lo, hi = self.pitch_limits
            self.gimbal_pitch = max(lo, min(hi, self.gimbal_pitch + self._rates[0] * dt))
            yaw = self.gimbal_yaw + self._rates[1] * dt
            self.gimbal_yaw = wrap180(yaw) if self.yaw_limits is None else max(self.yaw_limits[0],
                                                                                min(self.yaw_limits[1], yaw))

            boot_ms = int(t * 1000) & 0xFFFFFFFF
            if now - last_hb >= 1.0:
                self.conn.mav.heartbeat_send(
                    mav.MAV_TYPE_QUADROTOR, mav.MAV_AUTOPILOT_ARDUPILOTMEGA,
                    mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mav.MAV_MODE_FLAG_SAFETY_ARMED,
                    4, mav.MAV_STATE_ACTIVE)  # custom_mode 4 = GUIDED (ArduCopter)
                last_hb = now

            a = self.speed / self.radius * t
            north, east = self.radius * math.cos(a), self.radius * math.sin(a)
            vn, ve = -self.speed * math.sin(a), self.speed * math.cos(a)
            yaw = math.atan2(ve, vn)
            roll = math.atan(self.speed ** 2 / (self.radius * 9.81))
            lat0, lon0, amsl0 = self.home
            lat = lat0 + math.degrees(north / 6_378_137.0)
            lon = lon0 + math.degrees(east / (6_378_137.0 * math.cos(math.radians(lat0))))
            self.conn.mav.global_position_int_send(
                boot_ms, int(lat * 1e7), int(lon * 1e7), int((amsl0 + self.altitude) * 1000),
                int(self.altitude * 1000), int(vn * 100), int(ve * 100), 0,
                int(math.degrees(yaw) % 360 * 100))
            self.conn.mav.attitude_send(boot_ms, roll, 0.0, yaw, 0.0, 0.0, self.speed / self.radius)
            if now - last_status >= 0.5:  # battery drains ~3 %/min; healthy 3D fix
                pct = max(5.0, 100.0 - t / 18.0)
                self.conn.mav.sys_status_send(0, 0, 0, 250, int((14.8 + 1.6 * pct / 100.0) * 1000), 1450,
                                              int(pct), 0, 0, 0, 0, 0, 0)
                self.conn.mav.gps_raw_int_send(int(t * 1e6), 3, int(lat * 1e7), int(lon * 1e7),
                                               int((amsl0 + self.altitude) * 1000), 70, 110,
                                               int(self.speed * 100), int(math.degrees(yaw) % 360 * 100), 14)
                last_status = now
            q = euler_to_quat(0.0, math.radians(self.gimbal_pitch), math.radians(self.gimbal_yaw))
            flags = (mav.GIMBAL_DEVICE_FLAGS_ROLL_LOCK | mav.GIMBAL_DEVICE_FLAGS_PITCH_LOCK
                     | mav.GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME)
            nan = float("nan")
            self.conn.mav.gimbal_device_attitude_status_send(0, 0, boot_ms, flags, q, nan, nan, nan, 0)
            self._stop.wait(1.0 / self.rate_hz)

    def _drain(self) -> None:
        mav = self._mav
        while True:
            try:
                msg = self.conn.recv_match(type="COMMAND_LONG", blocking=False)
            except OSError:
                return
            if msg is None:
                return
            if msg.command == mav.MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW:
                pr = 0.0 if math.isnan(msg.param3) else msg.param3
                yr = 0.0 if math.isnan(msg.param4) else msg.param4
                self._rates, self._rate_t = (pr, yr), time.monotonic()
                self.gimbal_commands += 1
            self.conn.mav.command_ack_send(msg.command, mav.MAV_RESULT_ACCEPTED)
