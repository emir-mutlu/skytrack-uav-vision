from __future__ import annotations

from ..config import DetectorConfig
from .base import Detector
from .classic import ClassicTargetDetector


def build_detector(cfg: DetectorConfig) -> Detector:
    if cfg.kind == "classic":
        return ClassicTargetDetector(cfg)
    if cfg.kind == "yolo":
        from .yolo import YoloDetector  # lazy: ultralytics/torch are optional

        return YoloDetector(cfg)
    raise ValueError(f"Unknown detector kind: {cfg.kind}")


__all__ = ["Detector", "ClassicTargetDetector", "build_detector"]
