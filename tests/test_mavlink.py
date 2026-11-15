"""Vehicle integration: geolocation maths, gimbal controller, and live MAVLink loops
against the built-in simulated vehicle (real pymavlink over local TCP)."""

import json
import math
import time

import pytest

from skytrack.config import GimbalConfig, MavlinkConfig, load_config
from skytrack.geo import geolocate
from skytrack.gimbal import GimbalController

HOME = (-35.363261, 149.165230)
M_PER_DEG_LAT = math.pi * 6_378_137.0 / 180.0


def _geo(err, pitch=-90.0, yaw=0.0, veh_yaw=0.0, alt=100.0, hfov=90.0, aspect=1.0):
    return geolocate(err, *HOME, alt, (0.0, 0.0, math.radians(veh_yaw)), pitch, yaw, hfov, aspect)


# ---------------------------------------------------------------- geolocation
def test_nadir_centre_is_directly_below():
    g = _geo((0.0, 0.0))
    assert g.north_m == pytest.approx(0, abs=1e-6) and g.east_m == pytest.approx(0, abs=1e-6)
    assert (g.lat, g.lon) == pytest.approx(HOME)
    assert g.slant_range_m == pytest.approx(100.0)


def test_nadir_image_edges_follow_fov_and_heading():
    # HFOV 90 deg -> right image edge is 45 deg off nadir -> 100 m at 100 m AGL
    g = _geo((1.0, 0.0))
    assert (g.north_m, g.east_m) == pytest.approx((0.0, 100.0), abs=1e-6)
    g = _geo((1.0, 0.0), veh_yaw=90.0)  # nose east: image right points south
    assert (g.north_m, g.east_m) == pytest.approx((-100.0, 0.0), abs=1e-6)
    g = _geo((0.0, -1.0))  # top edge of a nadir image points forward (north)
    assert (g.north_m, g.east_m) == pytest.approx((100.0, 0.0), abs=1e-6)


def test_oblique_view_and_lat_lon_conversion():
    g = _geo((0.0, 0.0), pitch=-45.0)
    assert g.north_m == pytest.approx(100.0) and g.ground_range_m == pytest.approx(100.0)
    assert (g.lat - HOME[0]) * M_PER_DEG_LAT == pytest.approx(100.0, rel=1e-6)
    g = _geo((0.0, 0.0), pitch=-45.0, yaw=90.0)  # gimbal panned right -> east
    assert (g.north_m, g.east_m) == pytest.approx((0.0, 100.0), abs=1e-6)


def test_rays_at_or_above_horizon_return_none():
    assert _geo((0.0, 0.0), pitch=0.0) is None
    assert _geo((0.0, -0.5), pitch=-5.0) is None
    assert _geo((0.0, 0.0), alt=0.0) is None


# ---------------------------------------------------------------- gimbal controller
def test_gimbal_controller_signs_deadband_clamp_and_stop():
    ctl = GimbalController(GimbalConfig(kp=2.0, kd=0.0, max_rate_dps=30.0, deadband=0.05, rate_hz=1000),
                           hfov_deg=90.0, aspect=1.0)
    assert ctl.update((0.01, -0.01), 0.0) == (0.0, 0.0)  # inside deadband
    pitch, yaw = ctl.update((0.2, 0.2), 0.1)  # target right and below
    assert yaw > 0 and pitch < 0
    assert yaw == pytest.approx(2.0 * math.degrees(math.atan(0.2)))
    pitch, yaw = ctl.update((1.0, -1.0), 0.2)  # 45 deg error -> clamped
    assert (pitch, yaw) == (30.0, 30.0)
    assert ctl.update(None, 0.3) == (0.0, 0.0)  # target lost -> one stop command
    assert ctl.update(None, 0.4) is None


def test_gimbal_controller_rate_limit():
    ctl = GimbalController(GimbalConfig(rate_hz=10), hfov_deg=80.0, aspect=0.5625)
    assert ctl.update((0.3, 0.0), 0.00) is not None
    assert ctl.update((0.3, 0.0), 0.05) is None
    assert ctl.update((0.3, 0.0), 0.11) is not None


def test_quaternion_round_trip():
    from skytrack.simvehicle import euler_to_quat
    from skytrack.vehicle import quat_to_euler

    for rpy in [(0.0, -0.7, 0.3), (0.1, 0.2, -2.5), (0.0, -1.5, 1.0)]:
        assert quat_to_euler(euler_to_quat(*rpy)) == pytest.approx(rpy, abs=1e-9)


def test_mavlink_cli_overrides():
    from skytrack.cli import _overrides, build_parser

    args = build_parser().parse_args(["run", "-s", "x.mp4", "--mavlink", "tcp:127.0.0.1:5760",
                                      "--gimbal-follow", "--hfov", "62"])
    cfg = load_config(overrides=_overrides(args))
    assert cfg.mavlink.enabled and cfg.mavlink.connection == "tcp:127.0.0.1:5760"
    assert cfg.mavlink.gimbal.follow and cfg.camera.hfov_deg == 62


# ---------------------------------------------------------------- live MAVLink
def test_link_reads_telemetry_and_commands_gimbal():
    pytest.importorskip("pymavlink")
    from skytrack.simvehicle import SimVehicle
    from skytrack.vehicle import MavlinkLink

    sim = SimVehicle(connection="tcpin:127.0.0.1:5781").start()
    link = MavlinkLink(MavlinkConfig(enabled=True, connection="tcp:127.0.0.1:5781")).start()
    try:
        assert link.wait_connected(10)
        st = link.state
        assert st.armed and st.mode == "GUIDED"
        assert st.alt_rel == pytest.approx(100.0, abs=0.01)
        assert math.hypot((st.lat - HOME[0]) * M_PER_DEG_LAT, 0) < 200
        assert st.gimbal_pitch_deg == pytest.approx(-45.0, abs=0.5)
        for _ in range(10):
            link.send_gimbal_rates(-30.0, 20.0)
            time.sleep(0.05)
        time.sleep(0.3)
        assert sim.gimbal_commands >= 5
        assert link.state.gimbal_pitch_deg < -50 and link.state.gimbal_yaw_deg > 5
    finally:
        link.stop()
        sim.stop()


def test_closed_loop_gimbal_follow_keeps_target_centred(tmp_path):
    """Detector -> tracker -> lock -> MAVLink gimbal command -> simulated camera motion."""
    pytest.importorskip("pymavlink")
    from skytrack.pipeline import Pipeline
    from skytrack.simvehicle import SimVehicle
    from skytrack.synthetic import PanningScene

    sim = SimVehicle(connection="tcpin:127.0.0.1:5782").start()
    cfg = load_config("configs/demo.yaml", {
        "output": {"show": False, "telemetry": str(tmp_path / "t.jsonl")},
        "mavlink": {"enabled": True, "connection": "tcp:127.0.0.1:5782", "gimbal": {"follow": True}}})
    pipe = Pipeline(cfg)
    scene = PanningScene(width=960, height=540, hfov_deg=cfg.camera.hfov_deg, seed=7)
    try:
        summary = pipe.run((scene.step(sim.gimbal_pitch, sim.gimbal_yaw)[0] for _ in range(150)),
                           30.0, live=True)
    finally:
        sim.stop()
    assert summary["locks_acquired"] >= 1 and sim.gimbal_commands > 20
    recs = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    locked = [r for r in recs if r["lock"]["state"] == "LOCKED"]
    assert len(locked) > 50
    tail = locked[-20:]
    assert max(max(abs(e) for e in r["lock"]["error"]) for r in tail) < 0.1  # target held centred
    assert all(r["vehicle"]["connected"] for r in tail)
    assert all(r.get("target_geo") for r in tail)
    assert abs(sim.gimbal_pitch + 45.0) + abs(sim.gimbal_yaw) > 1.0  # the gimbal actually moved
