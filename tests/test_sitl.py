"""Integration tests against real ArduPilot SITL (the actual ArduCopter firmware built for
Linux). Skipped unless SKYTRACK_SITL_BIN points at an `arducopter` SITL binary; CI downloads
the official stable build from firmware.ardupilot.org.

    SKYTRACK_SITL_BIN=./arducopter pytest tests/test_sitl.py -s
"""

import os
import subprocess
import time

import pytest

from skytrack.config import MavlinkConfig

SITL_BIN = os.environ.get("SKYTRACK_SITL_BIN")
pytestmark = pytest.mark.skipif(not SITL_BIN, reason="set SKYTRACK_SITL_BIN to an ArduPilot SITL binary")

HOME = (-35.363261, 149.165230, 584.0, 353.0)  # lat, lon, alt AMSL, heading
# Quad + a servo gimbal on outputs 9 (tilt) and 10 (pan), driven by the gimbal manager.
PARAMS = """\
FRAME_CLASS 1
FRAME_TYPE 1
MNT1_TYPE 1
MNT1_PITCH_MIN -90
MNT1_PITCH_MAX 20
MNT1_YAW_MIN -180
MNT1_YAW_MAX 180
SERVO9_FUNCTION 7
SERVO10_FUNCTION 6
"""


@pytest.fixture(scope="module")
def sitl(tmp_path_factory):
    d = tmp_path_factory.mktemp("sitl")
    (d / "skytrack.parm").write_text(PARAMS)
    log_path = d / "sitl.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [SITL_BIN, "--model", "+", "--speedup", "1", "--defaults", "skytrack.parm", "-I0",
         "--home", ",".join(map(str, HOME))],
        cwd=d, stdout=log, stderr=subprocess.STDOUT,
    )
    time.sleep(1.0)
    yield proc
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    print("---- SITL log (tail) ----\n" + log_path.read_text(errors="replace")[-2000:])


def _wait(pred, timeout: float, step: float = 0.25) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return False


def test_sitl_telemetry_and_gimbal_rate_control(sitl):
    from skytrack.vehicle import MavlinkLink

    link = MavlinkLink(MavlinkConfig(enabled=True, connection="tcp:127.0.0.1:5760", connect_retries=30)).start()
    try:
        assert _wait(lambda: link.state.connected(), 60), "no heartbeat from SITL"

        def at_home():
            st = link.state
            return (st.has_position and abs(st.lat - HOME[0]) < 1e-3 and abs(st.lon - HOME[1]) < 1e-3)

        assert _wait(at_home, 120), f"no GPS position near home: {link.state.as_dict()}"
        st = link.state
        print("SITL state:", st.as_dict(), "rx:", dict(link.rx_counts))
        assert st.mode  # e.g. STABILIZE
        assert abs(st.alt_amsl - HOME[2]) < 20

        assert _wait(lambda: link.state.gimbal_pitch_deg is not None, 30), \
            f"no gimbal attitude from SITL mount; rx: {dict(link.rx_counts)}"
        p0 = link.state.gimbal_pitch_deg
        for _ in range(20):
            link.send_gimbal_rates(-20.0, 0.0)
            time.sleep(0.1)
        link.send_gimbal_rates(0.0, 0.0)
        time.sleep(0.5)
        p1 = link.state.gimbal_pitch_deg
        print(f"gimbal pitch {p0:+.1f} -> {p1:+.1f}, acks {link.acks}")
        assert link.acks.get(1000) == 0  # MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW accepted
        assert p1 < p0 - 10
    finally:
        link.stop()


def test_sitl_link_check_passes(sitl):
    from skytrack.linkcheck import run_link_check

    lines: list[str] = []
    ok = run_link_check(MavlinkConfig(enabled=True, connection="tcp:127.0.0.1:5762", connect_retries=30),
                        window_s=3.0, timeout_s=60.0, gimbal_test=True, out=lines.append)
    print("\n".join(lines))
    assert ok, "\n".join(lines)
