"""Bring-up diagnostics for a real flight controller (or SITL).

    skytrack link-check --mavlink COM5
    skytrack link-check --mavlink /dev/ttyACM0 --gimbal-test

Checks, in order: autopilot heartbeat, telemetry stream rates, GPS position, gimbal
attitude reporting, stream-request acknowledgement and (optionally) that the gimbal
really moves on a rate command. Each step prints PASS / WARN / FAIL with a hint, so the
first session on new hardware does not turn into guesswork.

The gimbal test only moves the camera gimbal (a small yaw sweep that returns to the
start). Run it with propellers removed, like any bench test.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

from .config import MavlinkConfig
from .vehicle import MavlinkLink, _mavutil

REQUIRED_STREAMS = {"GLOBAL_POSITION_INT": 1.0, "ATTITUDE": 1.0}
OPTIONAL_STREAMS = ("GIMBAL_DEVICE_ATTITUDE_STATUS", "MOUNT_STATUS", "GPS_RAW_INT", "SYS_STATUS")


class LinkCheck:
    def __init__(self, cfg: MavlinkConfig, out: Callable[[str], None] = print):
        self.cfg, self.out = cfg, out
        self.results: list[tuple[str, str]] = []

    def _report(self, status: str, msg: str) -> None:
        self.results.append((status, msg))
        self.out(f"  [{status:4s}] {msg}")

    @property
    def passed(self) -> bool:
        return not any(s == "FAIL" for s, _ in self.results)

    def run(self, window_s: float = 5.0, timeout_s: float = 30.0, gimbal_test: bool = False) -> bool:
        mav = _mavutil().mavlink
        self.out(f"SkyTrack link check -> {self.cfg.connection}")
        link = MavlinkLink(self.cfg).start()
        try:
            # 1) heartbeat -------------------------------------------------------------
            t_end = time.monotonic() + timeout_s
            while time.monotonic() < t_end and not link.state.connected(self.cfg.heartbeat_timeout):
                time.sleep(0.1)
            st = link.state
            if not st.connected(self.cfg.heartbeat_timeout):
                self._report("FAIL", f"no autopilot HEARTBEAT within {timeout_s:.0f} s - check port/baud, "
                                     "that the port runs MAVLink (SERIALx_PROTOCOL = 2) and the target system id")
                return False
            self._report("PASS", f"heartbeat: mode {st.mode}, {'ARMED' if st.armed else 'disarmed'}, "
                                 f"system {self.cfg.target_system}")

            # 2) streams ----------------------------------------------------------------
            c0 = Counter(link.rx_counts)
            time.sleep(window_s)
            rates = {k: v / window_s for k, v in (Counter(link.rx_counts) - c0).items()}
            for name, need in REQUIRED_STREAMS.items():
                hz = rates.get(name, 0.0)
                self._report("PASS" if hz >= need else "FAIL", f"{name:30s} {hz:5.1f} Hz")
            for name in OPTIONAL_STREAMS:
                if name in rates:
                    self._report("INFO", f"{name:30s} {rates[name]:5.1f} Hz")
            ack = link.acks.get(mav.MAV_CMD_SET_MESSAGE_INTERVAL)
            if ack is None:
                self._report("WARN", "no ACK for SET_MESSAGE_INTERVAL - stream rates may come from SRx_ params")
            elif ack != mav.MAV_RESULT_ACCEPTED:
                self._report("WARN", f"SET_MESSAGE_INTERVAL rejected ({_result_name(ack)})")

            # 3) position -------------------------------------------------------------
            st = link.state
            if st.has_position and not (abs(st.lat) < 1e-7 and abs(st.lon) < 1e-7):
                self._report("PASS", f"position {st.lat:.7f}, {st.lon:.7f}  alt {st.alt_rel:.1f} m rel / "
                                     f"{st.alt_amsl:.1f} m AMSL")
            else:
                self._report("WARN", "no valid position yet (GPS fix / EKF origin) - geolocation disabled")

            # 4) gimbal attitude ------------------------------------------------------
            if st.gimbal_pitch_deg is not None:
                self._report("PASS", f"gimbal attitude ({st.gimbal_source}): pitch {st.gimbal_pitch_deg:+.1f}, "
                                     f"yaw {st.gimbal_yaw_deg:+.1f}")
            else:
                self._report("WARN", "no gimbal attitude - configure a mount (MNT1_TYPE) and reboot; "
                                     "geolocation falls back to camera.default_pitch_deg")

            # 5) gimbal motion test ---------------------------------------------------
            if gimbal_test:
                self._gimbal_test(link, mav)
        finally:
            link.stop()
        self.out("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return self.passed

    def _gimbal_test(self, link: MavlinkLink, mav) -> None:
        def drive(yaw_rate: float, seconds: float) -> None:
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                link.send_gimbal_rates(0.0, yaw_rate)
                time.sleep(0.1)
            link.send_gimbal_rates(0.0, 0.0)
            time.sleep(0.5)

        y0 = link.state.gimbal_yaw_deg
        drive(15.0, 1.5)
        y1 = link.state.gimbal_yaw_deg
        drive(-15.0, 1.5)  # sweep back towards the start
        ack = link.acks.get(mav.MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW)
        if ack is None:
            self._report("FAIL", "no ACK for DO_GIMBAL_MANAGER_PITCHYAW - autopilot firmware without gimbal "
                                 "manager support (ArduPilot >= 4.3 / PX4 >= 1.13 required)")
            return
        if ack != mav.MAV_RESULT_ACCEPTED:
            self._report("FAIL", f"DO_GIMBAL_MANAGER_PITCHYAW rejected ({_result_name(ack)})")
            return
        self._report("PASS", "DO_GIMBAL_MANAGER_PITCHYAW accepted")
        if y0 is None or y1 is None:
            self._report("WARN", "command accepted but no gimbal attitude to confirm motion")
            return
        moved = abs((y1 - y0 + 180) % 360 - 180)
        self._report("PASS" if moved > 5.0 else "FAIL",
                     f"gimbal yaw moved {moved:.1f} deg on a 15 deg/s x 1.5 s command"
                     + ("" if moved > 5.0 else " - check mount mode, servo outputs and yaw limits"))


def _result_name(code: int) -> str:
    try:
        return _mavutil().mavlink.enums["MAV_RESULT"][code].name
    except (KeyError, AttributeError):
        return str(code)


def run_link_check(cfg: MavlinkConfig, window_s: float = 5.0, timeout_s: float = 30.0,
                   gimbal_test: bool = False, out: Callable[[str], None] = print) -> bool:
    return LinkCheck(cfg, out).run(window_s, timeout_s, gimbal_test)


__all__ = ["LinkCheck", "run_link_check"]
