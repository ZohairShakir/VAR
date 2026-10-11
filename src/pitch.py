"""Pitch keypoints, robust homography estimation, and temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics import YOLO

from src.config import PitchConfig


Frame = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SoccerPitch:
    """The 32-landmark pitch convention used by Roboflow Sports.

    Distances are expressed in metres. The default 120 x 70 m template must be
    replaced with verified stadium dimensions before interpreting Stage 4
    margins as survey-grade measurements.
    """

    width: float = 70.0
    length: float = 120.0
    penalty_box_width: float = 41.0
    penalty_box_length: float = 20.15
    goal_box_width: float = 18.32
    goal_box_length: float = 5.5
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @property
    def vertices(self) -> FloatArray:
        """Return the model's 32 pitch landmarks in label-index order."""

        width = self.width
        length = self.length
        penalty_width = self.penalty_box_width
        goal_width = self.goal_box_width
        penalty_length = self.penalty_box_length
        goal_length = self.goal_box_length
        spot = self.penalty_spot_distance
        circle = self.centre_circle_radius
        return np.asarray(
            [
                (0.0, 0.0),
                (0.0, (width - penalty_width) / 2),
                (0.0, (width - goal_width) / 2),
                (0.0, (width + goal_width) / 2),
                (0.0, (width + penalty_width) / 2),
                (0.0, width),
                (goal_length, (width - goal_width) / 2),
                (goal_length, (width + goal_width) / 2),
                (spot, width / 2),
                (penalty_length, (width - penalty_width) / 2),
                (penalty_length, (width - goal_width) / 2),
                (penalty_length, (width + goal_width) / 2),
                (penalty_length, (width + penalty_width) / 2),
                (length / 2, 0.0),
                (length / 2, width / 2 - circle),
                (length / 2, width / 2 + circle),
                (length / 2, width),
                (length - penalty_length, (width - penalty_width) / 2),
                (length - penalty_length, (width - goal_width) / 2),
                (length - penalty_length, (width + goal_width) / 2),
                (length - penalty_length, (width + penalty_width) / 2),
                (length - spot, width / 2),
                (length - goal_length, (width - goal_width) / 2),
                (length - goal_length, (width + goal_width) / 2),
                (length, 0.0),
                (length, (width - penalty_width) / 2),
                (length, (width - goal_width) / 2),
                (length, (width + goal_width) / 2),
                (length, (width + penalty_width) / 2),
                (length, width),
                (length / 2 - circle, width / 2),
                (length / 2 + circle, width / 2),
            ],
            dtype=np.float64,
        )

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return zero-based landmark pairs describing pitch markings."""

        one_based = (
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
            (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
            (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29),
            (17, 30),
        )
        return tuple((start - 1, end - 1) for start, end in one_based)


@dataclass(frozen=True, slots=True)
class PitchKeypoints:
    """One model prediction containing all 32 indexed landmarks."""

    xy: FloatArray
    confidence: FloatArray

    def __post_init__(self) -> None:
        if self.xy.shape != (32, 2):
            raise ValueError(
                "Expected keypoint coordinates with shape (32, 2), "
                f"got {self.xy.shape}."
            )
        if self.confidence.shape != (32,):
            raise ValueError(
                "Expected keypoint confidence with shape (32,), "
                f"got {self.confidence.shape}."
            )


@dataclass(frozen=True, slots=True)
class HomographyEstimate:
    """A validated image-to-pitch transform and its inlier correspondences."""

    matrix: FloatArray
    image_points: FloatArray
    pitch_points: FloatArray
    visible_count: int
    inlier_count: int
    median_error_m: float


class MappingStatus(str, Enum):
    """Quality/state attached to the transform used for the current frame."""

    FRESH = "fresh"
    SMOOTHED = "smoothed"
    RESET = "reset"
    HELD = "held"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class PitchMapping:
    """Pitch-mapping result for a frame."""

    homography: FloatArray | None
    status: MappingStatus
    keypoints: PitchKeypoints | None
    estimate: HomographyEstimate | None

    @property
    def is_available(self) -> bool:
        return self.homography is not None

    @property
    def is_fresh(self) -> bool:
        return self.status in {
            MappingStatus.FRESH,
            MappingStatus.SMOOTHED,
            MappingStatus.RESET,
        }


def project_points(
    points_xy: npt.ArrayLike,
    homography: npt.ArrayLike,
    *,
    epsilon: float = 1e-9,
) -> FloatArray:
    """Project ``(N, 2)`` points through a 3 x 3 homography."""

    points = np.asarray(points_xy, dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3).")
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError("points_xy and homography must contain finite values.")
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    projected = homogeneous @ matrix.T
    scale = projected[:, 2]
    if np.any(np.abs(scale) <= epsilon):
        raise ValueError("At least one point maps to infinity.")
    return projected[:, :2] / scale[:, None]


def _normalise_homography(matrix: npt.ArrayLike) -> FloatArray:
    """Normalize projective scale and reject singular/non-finite matrices."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Homography must be a finite 3 x 3 matrix.")
    if abs(float(np.linalg.det(value))) <= 1e-12:
        raise ValueError("Homography is singular.")
    scale = float(value[2, 2])
    if abs(scale) <= 1e-12:
        scale = float(np.linalg.norm(value))
    if abs(scale) <= 1e-12:
        raise ValueError("Homography has zero scale.")
    value = value / scale
    if value[2, 2] < 0:
        value = -value
    return value


class PitchKeypointDetector:
    """Run the pretrained 32-landmark Ultralytics pose model."""

    def __init__(self, config: PitchConfig) -> None:
        self.config = config
        model_path = Path(config.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pitch model not found: {model_path}. Download the published "
                "Roboflow Sports weights with: gdown "
                "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf -O "
                "data/weights/football-pitch-detection.pt"
            )
        try:
            self.model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load pitch model {model_path}: {exc}"
            ) from exc

    def detect(self, frame: Frame) -> PitchKeypoints | None:
        """Return the most confident pitch prediction, or ``None``."""

        kwargs: dict[str, Any] = {
            "source": frame,
            "imgsz": self.config.image_size,
            "conf": self.config.detection_confidence,
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        try:
            result = self.model.predict(**kwargs)[0]
        except Exception as exc:
            raise RuntimeError(f"Pitch-keypoint inference failed: {exc}") from exc
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None

        box_confidence = result.boxes.conf.detach().cpu().numpy()
        index = int(np.argmax(box_confidence)) if len(box_confidence) else 0
        xy = result.keypoints.xy[index].detach().cpu().numpy().astype(np.float64)
        if result.keypoints.conf is None:
            confidence = np.ones(len(xy), dtype=np.float64)
        else:
            confidence = (
                result.keypoints.conf[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        if xy.shape[0] != 32:
            raise ValueError(
                f"Pitch model returned {xy.shape[0]} landmarks; this pipeline expects 32."
            )
        return PitchKeypoints(xy=xy, confidence=confidence)


class HomographyEstimator:
    """Fit and validate an image-to-pitch homography with RANSAC."""

    def __init__(self, config: PitchConfig, pitch: SoccerPitch | None = None) -> None:
        self.config = config
        self.pitch = pitch or SoccerPitch()

    def estimate(self, keypoints: PitchKeypoints | None) -> HomographyEstimate | None:
        """Estimate a transform or return ``None`` when evidence is inadequate."""

        if keypoints is None:
            return None
        finite = np.isfinite(keypoints.xy).all(axis=1) & np.isfinite(
            keypoints.confidence
        )
        visible = finite & (
            keypoints.confidence >= self.config.keypoint_confidence
        )
        visible_count = int(np.count_nonzero(visible))
        if visible_count < self.config.min_visible_keypoints:
            return None

        image_points = keypoints.xy[visible]
        pitch_points = self.pitch.vertices[visible]
        matrix, inlier_mask = cv2.findHomography(
            image_points,
            pitch_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_reprojection_threshold_m,
        )
        if matrix is None or inlier_mask is None:
            return None

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < self.config.min_inliers:
            return None
        if inlier_count / visible_count < self.config.min_inlier_ratio:
            return None
        try:
            matrix = _normalise_homography(matrix)
            projected = project_points(image_points[inliers], matrix)
        except (ValueError, np.linalg.LinAlgError):
            return None
        errors = np.linalg.norm(projected - pitch_points[inliers], axis=1)
        median_error = float(np.median(errors))
        if median_error > self.config.max_median_reprojection_error_m:
            return None
        return HomographyEstimate(
            matrix=matrix,
            image_points=image_points[inliers],
            pitch_points=pitch_points[inliers],
            visible_count=visible_count,
            inlier_count=inlier_count,
            median_error_m=median_error,
        )


class HomographySmoother:
    """EMA-smooth compatible transforms and briefly bridge missed detections."""

    def __init__(self, config: PitchConfig) -> None:
        self.config = config
        self._matrix: FloatArray | None = None
        self._stale_frames = 0

    def reset(self) -> None:
        self._matrix = None
        self._stale_frames = 0

    def update(
        self,
        estimate: HomographyEstimate | None,
    ) -> tuple[FloatArray | None, MappingStatus]:
        """Update state and return the best transform for the current frame."""

        if estimate is None:
            if self._matrix is None:
                return None, MappingStatus.UNAVAILABLE
            self._stale_frames += 1
            if self._stale_frames <= self.config.max_stale_frames:
                return self._matrix.copy(), MappingStatus.HELD
            self.reset()
            return None, MappingStatus.UNAVAILABLE

        candidate = estimate.matrix
        self._stale_frames = 0
        if self._matrix is None:
            self._matrix = candidate.copy()
            return self._matrix.copy(), MappingStatus.FRESH

        try:
            previous_projection = project_points(
                estimate.image_points,
                self._matrix,
            )
            previous_error = float(
                np.median(
                    np.linalg.norm(
                        previous_projection - estimate.pitch_points,
                        axis=1,
                    )
                )
            )
        except ValueError:
            previous_error = float("inf")

        # On a large pan, zoom, cut, or stale transform, adopt the new robust
        # estimate immediately instead of smearing incompatible cameras.
        if previous_error > self.config.max_previous_error_m:
            self._matrix = candidate.copy()
            return self._matrix.copy(), MappingStatus.RESET

        alpha = self.config.smoothing_alpha
        blended = _normalise_homography(
            (1.0 - alpha) * self._matrix + alpha * candidate
        )
        try:
            smoothed_projection = project_points(estimate.image_points, blended)
            smoothed_error = float(
                np.median(
                    np.linalg.norm(
                        smoothed_projection - estimate.pitch_points,
                        axis=1,
                    )
                )
            )
        except ValueError:
            smoothed_error = float("inf")

        if smoothed_error > self.config.max_smoothed_error_m:
            self._matrix = candidate.copy()
            return self._matrix.copy(), MappingStatus.RESET
        self._matrix = blended
        return self._matrix.copy(), MappingStatus.SMOOTHED


class PitchMapper:
    """Orchestrate keypoint inference, robust fitting, and smoothing."""

    def __init__(
        self,
        config: PitchConfig,
        pitch: SoccerPitch | None = None,
    ) -> None:
        self.config = config
        self.pitch = pitch or SoccerPitch()
        self.detector = PitchKeypointDetector(config)
        self.estimator = HomographyEstimator(config, self.pitch)
        self.smoother = HomographySmoother(config)

    def reset(self) -> None:
        """Forget the preceding shot's transform."""

        self.smoother.reset()

    def map_frame(self, frame: Frame) -> PitchMapping:
        """Compute the best image-to-pitch mapping for one frame."""

        keypoints = self.detector.detect(frame)
        estimate = self.estimator.estimate(keypoints)
        homography, status = self.smoother.update(estimate)
        return PitchMapping(
            homography=homography,
            status=status,
            keypoints=keypoints,
            estimate=estimate,
        )
