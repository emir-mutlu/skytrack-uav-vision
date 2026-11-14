from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class Detector(ABC):
    """A detector maps a BGR frame to a list of detections."""

    name: str = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def warmup(self, shape: tuple[int, int, int] = (640, 640, 3)) -> None:
        self.detect(np.zeros(shape, dtype=np.uint8))
