"""Typed configuration loaded from YAML mission profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorConfig:
    kind: str = "classic"  # "classic" (contour/shape based) | "yolo" (deep learning)
    # --- YOLO (Ultralytics) ---
    model: str = "yolo11n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    classes: list[int] | None = None
    device: str | None = None  # "cpu", "0" (cuda), "mps" ...
    half: bool = False
    # --- Classic shape detector (modernised 2015 PyImageSearch approach) ---
    min_side: int = 15
    aspect_min: float = 0.8
    aspect_max: float = 1.2
    min_solidity: float = 0.9
    clahe: bool = True


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.25
    max_age: int = 15  # frames a track survives without a matching detection
    min_hits: int = 3  # detections required before a track is confirmed
    history: int = 30  # trail length kept per track


@dataclass
class LockConfig:
    enabled: bool = True
    # Lock zone as fractions of the frame (x1, y1, x2, y2).
    zone: tuple[float, float, float, float] = (0.25, 0.10, 0.75, 0.90)
    min_size_frac: float = 0.03  # target width or height must exceed this fraction of the frame
    acquire_seconds: float = 1.0  # continuous time in zone before lock is declared
    lost_grace_seconds: float = 0.5  # tolerated dropout before lock is lost
    target_labels: list[str] | None = None  # None = any class
    require_operator_confirm: bool = True  # human-in-the-loop: operator designates target


@dataclass
class CameraConfig:
    hfov_deg: float = 80.0  # horizontal field of view of the lens
    stabilized: bool = True  # 3-axis gimbal holds roll/pitch level (ignore vehicle roll/pitch)
    # Used when no gimbal attitude is reported (fixed mount, or before the first report):
    default_pitch_deg: float = -90.0  # -90 = nadir, 0 = horizon
    default_yaw_deg: float = 0.0  # relative to the vehicle nose


@dataclass
class TerrainConfig:
    source: str = "flat"  # "flat" (ground at home altitude) | "hgt" (SRTM / NASADEM tiles)
    path: str | None = None  # directory with *.hgt tiles when source = "hgt"


@dataclass
class GimbalConfig:
    follow: bool = False  # keep the designated target centred by commanding gimbal rates
    kp: float = 2.0  # (deg/s) per degree of angular error
    ki: float = 0.0
    kd: float = 0.05
    max_rate_dps: float = 60.0
    deadband: float = 0.02  # normalised image error ignored around the centre
    rate_hz: float = 20.0  # max command rate sent to the autopilot


@dataclass
class MavlinkConfig:
    enabled: bool = False
    # pymavlink connection string: udpin:0.0.0.0:14550 (SITL / telemetry radio via MAVProxy),
    # tcp:127.0.0.1:5760, COM5 / /dev/ttyACM0 (USB), /dev/ttyTHS1 (Jetson UART) ...
    connection: str = "udpin:0.0.0.0:14550"
    baud: int = 57600
    source_system: int = 1
    source_component: int = 191  # MAV_COMP_ID_ONBOARD_COMPUTER
    target_system: int = 1
    target_component: int = 1  # autopilot (acts as gimbal manager)
    stream_hz: float = 10.0  # requested rate for position / attitude / gimbal messages
    heartbeat_timeout: float = 3.0
    connect_retries: int = 10  # TCP connect attempts (1 s apart) - SITL may still be booting
    geolocate: bool = True
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)


@dataclass
class OutputConfig:
    show: bool = True
    save_video: str | None = None
    telemetry: str | None = None  # JSONL path
    draw_trails: bool = True


@dataclass
class Config:
    profile: str = "default"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    lock: LockConfig = field(default_factory=LockConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    valid = {f.name: f for f in fields(dc)}
    for key, value in (data or {}).items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _merge(cfg, overrides)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.detector.kind not in {"classic", "yolo"}:
        raise ValueError(f"detector.kind must be 'classic' or 'yolo', got {cfg.detector.kind!r}")
    x1, y1, x2, y2 = cfg.lock.zone
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"lock.zone must be normalised (x1<x2, y1<y2 in [0,1]), got {cfg.lock.zone}")
    if cfg.terrain.source not in {"flat", "hgt"}:
        raise ValueError(f"terrain.source must be 'flat' or 'hgt', got {cfg.terrain.source!r}")
    if cfg.terrain.source == "hgt" and not cfg.terrain.path:
        raise ValueError("terrain.path is required when terrain.source = 'hgt'")
    if not 0 < cfg.detector.conf < 1:
        raise ValueError("detector.conf must be in (0, 1)")
