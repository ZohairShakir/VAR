"""Human pose estimation for tracked players in a selected frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from ultralytics import YOLO

from src.config import PoseConfig


Frame = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# IFAB Law 11 considers head, body, and feet, but not hands or arms. COCO does
# not provide armpit boundaries, so shoulder centres are retained as a torso
# approximation while elbows and wrists are explicitly excluded.
OFFSIDE_RELEVANT_INDICES: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16
)
EXCLUDED_ARM_INDICES: tuple[int, ...] = (7, 8, 9, 10)

LEGAL_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True, slots=True)
class PlayerPose:
    """COCO keypoints associated with one tracked player."""

    tracker_id: int
    box_xyxy: FloatArray
    keypoints_xy: FloatArray
    confidence: FloatArray

    def __post_init__(self) -> None:
        if self.box_xyxy.shape != (4,):
            raise ValueError("box_xyxy must have shape (4,).")
        if self.keypoints_xy.shape != (17, 2):
            raise ValueError("keypoints_xy must have shape (17, 2).")
        if self.confidence.shape != (17,):
            raise ValueError("confidence must have shape (17,).")

    def legal_mask(self, threshold: float) -> npt.NDArray[np.bool_]:
        """Return confident head, torso, leg, and foot keypoints."""

        mask = np.zeros(17, dtype=bool)
        mask[list(OFFSIDE_RELEVANT_INDICES)] = True
        return (
            mask
            & np.isfinite(self.keypoints_xy).all(axis=1)
            & np.isfinite(self.confidence)
            & (self.confidence >= threshold)
        )


@dataclass(frozen=True, slots=True)
class _Crop:
    tracker_id: int
    box_xyxy: FloatArray
    image: Frame
    offset_xy: tuple[int, int]
    expected_center_xy: tuple[float, float]


class PoseEstimator:
    """Estimate a separate pose inside each tracked player crop."""

    def __init__(self, config: PoseConfig) -> None:
        self.config = config
        model_path = Path(config.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}. Download the official "
                "Ultralytics checkpoint by running YOLO('yolo11n-pose.pt') "
                "inside data/weights or provide --pose-model."
            )
        try:
            self.model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load pose model {model_path}: {exc}"
            ) from exc

    def estimate(
        self,
        frame: Frame,
        boxes_xyxy: npt.ArrayLike,
        tracker_ids: npt.ArrayLike,
    ) -> list[PlayerPose]:
        """Return poses for valid crops, preserving track association."""

        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
        ids = np.asarray(tracker_ids, dtype=int)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes_xyxy must have shape (N, 4).")
        if len(boxes) != len(ids):
            raise ValueError("Expected one tracker ID for each box.")
        crops = [
            crop
            for box, tracker_id in zip(boxes, ids, strict=True)
            if (crop := self._make_crop(frame, box, int(tracker_id))) is not None
        ]
        if not crops:
            return []

        kwargs: dict[str, Any] = {
            "source": [crop.image for crop in crops],
            "imgsz": self.config.image_size,
            "conf": self.config.detection_confidence,
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        try:
            results = self.model.predict(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Human-pose inference failed: {exc}") from exc

        poses: list[PlayerPose] = []
        for crop, result in zip(crops, results, strict=True):
            pose = self._pose_from_result(crop, result)
            if pose is not None:
                poses.append(pose)
        return poses

    def _make_crop(
        self,
        frame: Frame,
        box_xyxy: FloatArray,
        tracker_id: int,
    ) -> _Crop | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 2 or box_height <= 4:
            return None
        padding_x = box_width * self.config.crop_padding_ratio
        padding_y = box_height * self.config.crop_padding_ratio
        crop_x1 = int(np.clip(np.floor(x1 - padding_x), 0, width))
        crop_y1 = int(np.clip(np.floor(y1 - padding_y), 0, height))
        crop_x2 = int(np.clip(np.ceil(x2 + padding_x), 0, width))
        crop_y2 = int(np.clip(np.ceil(y2 + padding_y), 0, height))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        expected_center = (
            float((x1 + x2) / 2 - crop_x1),
            float((y1 + y2) / 2 - crop_y1),
        )
        return _Crop(
            tracker_id=tracker_id,
            box_xyxy=np.asarray(box_xyxy, dtype=np.float64),
            image=image,
            offset_xy=(crop_x1, crop_y1),
            expected_center_xy=expected_center,
        )

    def _pose_from_result(self, crop: _Crop, result: Any) -> PlayerPose | None:
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float64)
        centers = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)
        )
        expected = np.asarray(crop.expected_center_xy, dtype=np.float64)
        diagonal = max(float(np.hypot(*crop.image.shape[:2])), 1.0)
        normalized_distance = np.linalg.norm(centers - expected, axis=1) / diagonal
        selection_score = confidences - normalized_distance
        index = int(np.argmax(selection_score))

        keypoints = (
            result.keypoints.xy[index].detach().cpu().numpy().astype(np.float64)
        )
        if keypoints.shape != (17, 2):
            raise ValueError(
                f"Pose model returned keypoint shape {keypoints.shape}; expected (17, 2)."
            )
        if result.keypoints.conf is None:
            keypoint_confidence = np.ones(17, dtype=np.float64)
        else:
            keypoint_confidence = (
                result.keypoints.conf[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        keypoints[:, 0] += crop.offset_xy[0]
        keypoints[:, 1] += crop.offset_xy[1]
        return PlayerPose(
            tracker_id=crop.tracker_id,
            box_xyxy=crop.box_xyxy.copy(),
            keypoints_xy=keypoints,
            confidence=keypoint_confidence,
        )
