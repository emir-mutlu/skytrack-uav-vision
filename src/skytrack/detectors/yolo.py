"""Deep-learning detector backed by Ultralytics YOLO (v8 / 11).

Works with COCO-pretrained weights out of the box (person, car, truck, airplane ...)
or with custom weights fine-tuned on aerial datasets (e.g. VisDrone for recon,
an anti-UAV dataset for drone-vs-drone detection). ONNX / TensorRT / OpenVINO
exports produced by `yolo export` load through the same interface.
"""

from __future__ import annotations

import numpy as np

from ..config import DetectorConfig
from ..types import Detection
from .base import Detector


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, cfg: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "YOLO detector requires the optional extra: pip install -e .[yolo]"
            ) from exc
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.names: dict[int, str] = dict(self.model.names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=self.cfg.classes,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(tuple(map(float, b)), float(s), int(c), self.names.get(int(c), str(c)))
            for b, s, c in zip(xyxy, conf, cls, strict=True)
        ]
