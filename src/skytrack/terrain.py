"""Terrain models for geolocation.

* FlatTerrain - constant ground elevation (default: the home altitude).
* HgtTerrain  - SRTM / NASADEM `.hgt` tiles (1" or 3" arc-second grids), read with NumPy
                only: no GDAL or rasterio needed on the flight computer.

`intersect` marches a camera ray across the terrain and refines the first crossing by
bisection, so targets on slopes, hillsides and valleys are placed where the line of
sight actually meets the ground instead of on a fictional flat plane.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6_378_137.0
HGT_VOID = -32768


class FlatTerrain:
    def __init__(self, elevation_m: float):
        self.elevation_m = float(elevation_m)

    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.full(np.shape(lat), self.elevation_m, dtype=np.float64)


class FunctionTerrain:
    """Analytic terrain `f(lat, lon) -> elevation`, handy for tests and what-if studies."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.fn = fn

    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(lat, np.float64), np.asarray(lon, np.float64)), np.float64)


def hgt_name(lat: float, lon: float) -> str:
    la, lo = math.floor(lat), math.floor(lon)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}.hgt"


class HgtTerrain:
    """Directory of SRTM `.hgt` tiles (e.g. `N41E028.hgt`). Tiles load lazily and are cached.
    Missing tiles or void cells give NaN, which callers treat as 'no data'."""

    _NAME = re.compile(r"^([NS])(\d{2})([EW])(\d{3})\.hgt$", re.IGNORECASE)

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"DEM directory not found: {self.dir}")
        self._files = {p.name.upper(): p for p in self.dir.iterdir() if self._NAME.match(p.name)}
        self._cache: dict[str, np.ndarray | None] = {}

    @property
    def tiles(self) -> list[str]:
        return sorted(self._files)

    def _tile(self, name: str) -> np.ndarray | None:
        if name not in self._cache:
            path = self._files.get(name.upper())
            if path is None:
                self._cache[name] = None
            else:
                raw = np.fromfile(path, dtype=">i2")
                n = int(round(math.sqrt(raw.size)))
                if n * n != raw.size:
                    raise ValueError(f"{path.name}: not a square .hgt grid ({raw.size} samples)")
                grid = raw.reshape(n, n).astype(np.float64)
                grid[grid == HGT_VOID] = np.nan
                self._cache[name] = grid
        return self._cache[name]

    def elevation(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        lon = np.atleast_1d(np.asarray(lon, np.float64))
        out = np.full(lat.shape, np.nan)
        la0, lo0 = np.floor(lat), np.floor(lon)
        for a, o in set(zip(la0.tolist(), lo0.tolist(), strict=True)):
            grid = self._tile(hgt_name(a, o))
            if grid is None:
                continue
            m = (la0 == a) & (lo0 == o)
            n = grid.shape[0] - 1
            r = (a + 1.0 - lat[m]) * n  # rows run north -> south
            c = (lon[m] - o) * n  # columns run west -> east
            r0 = np.clip(np.floor(r).astype(int), 0, n - 1)
            c0 = np.clip(np.floor(c).astype(int), 0, n - 1)
            fr, fc = r - r0, c - c0
            out[m] = (grid[r0, c0] * (1 - fr) * (1 - fc) + grid[r0, c0 + 1] * (1 - fr) * fc
                      + grid[r0 + 1, c0] * fr * (1 - fc) + grid[r0 + 1, c0 + 1] * fr * fc)
        return out


def offset_latlon(lat: float, lon: float, north: np.ndarray, east: np.ndarray):
    lat2 = lat + np.degrees(north / EARTH_RADIUS_M)
    lon2 = lon + np.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat2, lon2


def intersect(
    lat: float,
    lon: float,
    cam_alt_amsl: float,
    ray_ned: np.ndarray,
    terrain,
    max_range_m: float = 5000.0,
    step_m: float = 10.0,
    iterations: int = 30,
) -> tuple[float, float, float, float] | None:
    """First intersection of a NED unit ray from the camera with the terrain.

    Returns (north_m, east_m, ground_elevation_m, slant_range_m) or None if the ray leaves
    `max_range_m`, points upwards, or crosses a region without elevation data."""
    if ray_ned[2] <= 1e-6:
        return None
    horiz = math.hypot(ray_ned[0], ray_ned[1])
    t_max = max_range_m / horiz if horiz > 1e-9 else cam_alt_amsl + 1.0e4
    t_max = min(t_max, 1.0e5)
    n = int(min(4000, max(16, math.ceil(t_max / step_m)))) + 1
    t = np.linspace(0.0, t_max, n)
    la, lo = offset_latlon(lat, lon, ray_ned[0] * t, ray_ned[1] * t)
    above = (cam_alt_amsl - ray_ned[2] * t) - terrain.elevation(la, lo)
    if np.isnan(above[0]) or above[0] <= 0:
        return None
    crossing = np.flatnonzero(~(above > 0))  # first sample at/below ground or without data
    if crossing.size == 0:
        return None
    k = int(crossing[0])
    if np.isnan(above[k]):
        return None
    lo_t, hi_t = float(t[k - 1]), float(t[k])
    for _ in range(iterations):
        mid = 0.5 * (lo_t + hi_t)
        a, o = offset_latlon(lat, lon, np.array([ray_ned[0] * mid]), np.array([ray_ned[1] * mid]))
        h = float(terrain.elevation(a, o)[0])
        if np.isnan(h):
            return None
        if cam_alt_amsl - ray_ned[2] * mid > h:
            lo_t = mid
        else:
            hi_t = mid
    tt = 0.5 * (lo_t + hi_t)
    return float(ray_ned[0] * tt), float(ray_ned[1] * tt), float(cam_alt_amsl - ray_ned[2] * tt), tt
