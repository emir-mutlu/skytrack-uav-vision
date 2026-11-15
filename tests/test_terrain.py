"""DEM-aware geolocation: ray marching against analytic terrain and SRTM .hgt tiles."""

import math

import numpy as np
import pytest

from skytrack.config import load_config
from skytrack.geo import geolocate
from skytrack.terrain import FlatTerrain, FunctionTerrain, HgtTerrain, hgt_name

HOME = (-35.363261, 149.165230)
GROUND = 584.0
M_PER_DEG_LAT = math.pi * 6_378_137.0 / 180.0


def _geo(terrain, pitch=-45.0, yaw=0.0, cam_alt=GROUND + 100.0, err=(0.0, 0.0)):
    return geolocate(err, *HOME, cam_alt - GROUND, (0.0, 0.0, 0.0), pitch, yaw, 90.0, 1.0,
                     terrain=terrain, cam_alt_amsl=cam_alt)


def test_flat_dem_matches_flat_model():
    dem = _geo(FlatTerrain(GROUND))
    flat = geolocate((0.0, 0.0), *HOME, 100.0, (0.0, 0.0, 0.0), -45.0, 0.0, 90.0, 1.0)
    assert dem.method == "dem" and flat.method == "flat"
    assert dem.north_m == pytest.approx(flat.north_m, abs=0.01)
    assert dem.ground_alt_m == pytest.approx(GROUND, abs=0.01)


def test_upslope_target_is_closer_than_flat_estimate():
    # ground rises 0.5 m per metre northwards: 100 - d = 0.5 d  ->  d = 66.7 m
    slope = FunctionTerrain(lambda la, lo: GROUND + 0.5 * (la - HOME[0]) * M_PER_DEG_LAT)
    g = _geo(slope)
    assert g.north_m == pytest.approx(200.0 / 3.0, abs=0.05)
    assert g.ground_alt_m == pytest.approx(GROUND + 100.0 / 3.0, abs=0.05)


def test_valley_target_is_farther_and_ridge_blocks_line_of_sight():
    valley = FunctionTerrain(lambda la, lo: np.full(np.shape(la), GROUND - 50.0))
    assert _geo(valley).north_m == pytest.approx(150.0, abs=0.05)
    # a 90 m ridge 60-70 m north intercepts a ray that would land at 100 m on flat ground
    ridge = FunctionTerrain(lambda la, lo: np.where(
        ((la - HOME[0]) * M_PER_DEG_LAT > 60) & ((la - HOME[0]) * M_PER_DEG_LAT < 70), GROUND + 90.0, GROUND))
    assert _geo(ridge).north_m == pytest.approx(60.0, abs=0.1)


def test_ray_above_horizon_has_no_dem_hit():
    assert _geo(FlatTerrain(GROUND), pitch=5.0) is None


def _write_hgt(path, n, fn):
    rows, cols = np.mgrid[0:n, 0:n]
    fn(rows, cols).astype(">i2").tofile(path)


def test_hgt_tile_naming_and_bilinear_lookup(tmp_path):
    assert hgt_name(-35.36, 149.16) == "S36E149.hgt"
    assert hgt_name(41.01, 28.97) == "N41E028.hgt"
    assert hgt_name(40.5, -3.7) == "N40W004.hgt"
    n = 11  # coarse 0.1-degree grid; real SRTM tiles are 1201 or 3601 samples wide
    _write_hgt(tmp_path / "S36E149.hgt", n, lambda r, c: 500 + 10 * r + c)
    dem = HgtTerrain(tmp_path)
    assert dem.tiles == ["S36E149.HGT"]
    # tile spans lat -36..-35 (row 0 = north edge -35), lon 149..150
    assert dem.elevation(np.array([-35.05]), np.array([149.0]))[0] == pytest.approx(505)
    assert dem.elevation(np.array([-35.5]), np.array([149.5]))[0] == pytest.approx(555)
    assert dem.elevation(np.array([-35.55]), np.array([149.25]))[0] == pytest.approx(500 + 55 + 2.5)
    assert np.isnan(dem.elevation(np.array([10.0]), np.array([10.0]))[0])  # no tile


def test_hgt_voids_fall_back_to_flat_model(tmp_path):
    _write_hgt(tmp_path / "S36E149.hgt", 11, lambda r, c: np.full(r.shape, -32768))
    g = _geo(HgtTerrain(tmp_path))
    assert g is not None and g.method == "flat"
    assert g.north_m == pytest.approx(100.0, abs=0.01)


def test_terrain_config_validation(tmp_path):
    with pytest.raises(ValueError):
        load_config(overrides={"terrain": {"source": "hgt"}})
    with pytest.raises(ValueError):
        load_config(overrides={"terrain": {"source": "lidar"}})
    cfg = load_config(overrides={"terrain": {"source": "hgt", "path": str(tmp_path)}})
    assert cfg.terrain.path == str(tmp_path)
