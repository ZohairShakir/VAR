"""Football-object detection backed by an Ultralytics YOLO model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import supervision as sv
from ultralytics import YOLO

from src.config import DetectorConfig


Frame = npt.NDArray[np.uint8]


def _normalise_name(name: str) -> str:
    """Return a model class name in a comparison-friendly form."""

    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def _names_as_dict(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
    """Normalise the two class-name containers supported by Ultralytics."""

    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


class FootballDetector:
    """Run YOLO and retain people/football-related detections only.

    The default COCO model has only ``person`` and ``sports ball`` classes. If
    custom weights expose ``player``, ``goalkeeper`` or ``referee``, this class
    preserves those more specific labels automatically.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        try:
            self.model = YOLO(config.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLO model {config.model_path!r}: {exc}"
            ) from exc
        self.class_names = _names_as_dict(self.model.names)

        wanted = {_normalise_name(name) for name in config.target_class_names}
        self.target_class_ids = sorted(
            class_id
            for class_id, name in self.class_names.items()
            if _normalise_name(name) in wanted
        )
        if not self.target_class_ids:
            available = ", ".join(sorted(self.class_names.values()))
            raise ValueError(
                "The selected model has no supported football classes. "
                f"Available classes: {available}"
            )

    def canonical_label(self, class_id: int) -> str:
        """Map a model-specific class to a stable display label."""

        raw_name = self.class_names.get(int(class_id), f"class-{class_id}")
        name = _normalise_name(raw_name)
        if name in {"sports ball", "ball", "football", "soccer ball"}:
            return "ball"
        return name

    def detect(self, frame: Frame) -> sv.Detections:
        """Detect relevant objects in one BGR frame.

        A low global threshold lets small balls reach post-processing. People
        are then filtered with their stricter, class-specific threshold.
        """

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3).")

        inference_confidence = min(
            self.config.person_confidence,
            self.config.ball_confidence,
        )
        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": inference_confidence,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "classes": self.target_class_ids,
            "verbose": False,
        }
        if self.config.device:
            predict_kwargs["device"] = self.config.device

        try:
            result = self.model.predict(**predict_kwargs)[0]
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc
        detections = sv.Detections.from_ultralytics(result)
        if detections.class_id is None or detections.confidence is None:
            return sv.Detections.empty()

        keep = np.fromiter(
            (
                confidence >= self._confidence_for(int(class_id))
                for class_id, confidence in zip(
                    detections.class_id,
                    detections.confidence,
                    strict=True,
                )
            ),
            dtype=bool,
            count=len(detections),
        )
        return detections[keep]

    def _confidence_for(self, class_id: int) -> float:
        """Return the class-specific confidence floor."""

        if self.canonical_label(class_id) == "ball":
            return self.config.ball_confidence
        return self.config.person_confidence
