"""Pixel -> ground geolocation for a gimballed (or fixed) UAV camera.

A pixel is turned into a viewing ray with a pinhole model derived from the lens HFOV,
rotated into the local North-East-Down frame using the gimbal and vehicle attitude,
and intersected with the ground: either a flat plane `alt_agl` metres below the camera,
or - when a terrain model (DEM) is supplied - the real terrain surface (see terrain.py).

Conventions (aerospace, right-handed, FRD / NED):
    camera frame  x = optical axis, y = image right, z = image down
    pitch         positive = up, -90 deg = nadir
    yaw           positive = clockwise seen from above (towards east from north)

The flat model is accurate to a few metres for ranges below ~2 km over level ground;
use a DEM for hills, slopes and valleys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float
    north_m: float  # offset from the vehicle
    east_m: float
    ground_range_m: float
    slant_range_m: float
    ground_alt_m: float | None = None  # terrain elevation AMSL at the target (DEM only)
    method: str = "flat"  # "flat" | "dem"


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def camera_ray(err: tuple[float, float], hfov_deg: float, aspect: float) -> np.ndarray:
    """Unit ray in the camera frame for a normalised image offset `err` in [-1, 1]
    (as produced by TargetLock.centre_error). `aspect` = height / width."""
    tan_h = math.tan(math.radians(hfov_deg) / 2.0)
    ray = np.array([1.0, err[0] * tan_h, err[1] * tan_h * aspect])
    return ray / np.linalg.norm(ray)


def camera_to_ned(
    vehicle_rpy: tuple[float, float, float],
    gimbal_pitch_deg: float,
    gimbal_yaw_deg: float,
    stabilized: bool = True,
) -> np.ndarray:
    """Rotation matrix camera -> NED. `vehicle_rpy` in radians, gimbal angles in degrees
    (yaw relative to the vehicle nose)."""
    roll, pitch, yaw = vehicle_rpy
    gp, gy = math.radians(gimbal_pitch_deg), math.radians(gimbal_yaw_deg)
    if stabilized:  # gimbal cancels vehicle roll/pitch, yaw follows the airframe
        return _rz(yaw + gy) @ _ry(gp)
    return _rz(yaw) @ _ry(pitch) @ _rx(roll) @ _rz(gy) @ _ry(gp)


def geolocate(
    err: tuple[float, float],
    lat: float,
    lon: float,
    alt_agl: float,
    vehicle_rpy: tuple[float, float, float],
    gimbal_pitch_deg: float,
    gimbal_yaw_deg: float,
    hfov_deg: float,
    aspect: float,
    stabilized: bool = True,
    max_range_m: float = 5000.0,
    terrain=None,
    cam_alt_amsl: float | None = None,
) -> GeoPoint | None:
    """Ground position of the pixel at normalised offset `err`, or None if the ray does
    not hit the ground within `max_range_m` (at/above the horizon or too shallow).

    With `terrain` (any object with `elevation(lat, lon)`) and `cam_alt_amsl`, the ray is
    intersected with the terrain surface; where the DEM has no data the flat model is used."""
    ray = camera_to_ned(vehicle_rpy, gimbal_pitch_deg, gimbal_yaw_deg, stabilized) @ camera_ray(
        err, hfov_deg, aspect
    )
    if terrain is not None and cam_alt_amsl is not None:
        from .terrain import intersect

        hit = intersect(lat, lon, cam_alt_amsl, ray, terrain, max_range_m)
        if hit is not None:
            north, east, ground_alt, slant = hit
            dlat = math.degrees(north / EARTH_RADIUS_M)
            dlon = math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
            return GeoPoint(lat + dlat, lon + dlon, north, east, math.hypot(north, east), slant,
                            ground_alt, "dem")
    if alt_agl <= 0.5:
        return None
    if ray[2] <= 1e-6:
        return None
    scale = alt_agl / ray[2]
    north, east = float(ray[0] * scale), float(ray[1] * scale)
    ground = math.hypot(north, east)
    if ground > max_range_m:
        return None
    dlat = math.degrees(north / EARTH_RADIUS_M)
    dlon = math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return GeoPoint(lat + dlat, lon + dlon, north, east, ground, float(scale))
