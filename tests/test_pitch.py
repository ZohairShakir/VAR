"""Synthetic tests for Stage 3 homography fitting and temporal state."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.config import PitchConfig
from src.pitch import (
    HomographyEstimator,
    HomographySmoother,
    MappingStatus,
    PitchKeypoints,
    SoccerPitch,
    project_points,
)


def _synthetic_keypoints(
    pitch_to_image: np.ndarray,
    visible_indices: np.ndarray,
    *,
    outlier_index: int | None = None,
) -> PitchKeypoints:
    pitch = SoccerPitch()
    image_points = project_points(pitch.vertices, pitch_to_image)
    confidence = np.zeros(32, dtype=np.float64)
    confidence[visible_indices] = 0.95
    if outlier_index is not None:
        image_points[outlier_index] += np.array([180.0, -130.0])
    return PitchKeypoints(image_points, confidence)


def test_ransac_recovers_synthetic_image_to_pitch_homography() -> None:
    """The geometry stage should reject one bad landmark and recover metres."""

    pitch_to_image = np.array(
        [
            [8.0, 0.8, 110.0],
            [0.2, 6.2, 75.0],
            [0.0015, 0.0020, 1.0],
        ]
    )
    visible = np.array([0, 1, 5, 8, 9, 13, 16, 20, 24, 29, 30, 31])
    keypoints = _synthetic_keypoints(
        pitch_to_image,
        visible,
        outlier_index=8,
    )
    config = replace(
        PitchConfig(),
        min_visible_keypoints=6,
        min_inliers=5,
        ransac_reprojection_threshold_m=0.25,
        max_median_reprojection_error_m=0.10,
    )

    estimate = HomographyEstimator(config).estimate(keypoints)

    assert estimate is not None
    assert estimate.visible_count == len(visible)
    assert estimate.inlier_count >= len(visible) - 1
    recovered = project_points(keypoints.xy[visible], estimate.matrix)
    errors = np.linalg.norm(recovered - SoccerPitch().vertices[visible], axis=1)
    # RANSAC's single outlier remains bad, but all other points are sub-mm here.
    assert float(np.median(errors)) < 1e-3


def test_estimator_gracefully_rejects_too_few_landmarks() -> None:
    matrix = np.array(
        [
            [8.0, 0.0, 100.0],
            [0.0, 6.0, 80.0],
            [0.001, 0.002, 1.0],
        ]
    )
    keypoints = _synthetic_keypoints(matrix, np.array([0, 13, 29]))

    assert HomographyEstimator(PitchConfig()).estimate(keypoints) is None


def test_smoother_bridges_brief_misses_then_expires() -> None:
    pitch_to_image = np.array(
        [
            [8.0, 0.2, 100.0],
            [0.1, 6.0, 80.0],
            [0.001, 0.001, 1.0],
        ]
    )
    visible = np.array([0, 1, 5, 8, 9, 13, 16, 20, 24, 29, 30, 31])
    config = replace(PitchConfig(), max_stale_frames=2)
    estimate = HomographyEstimator(config).estimate(
        _synthetic_keypoints(pitch_to_image, visible)
    )
    assert estimate is not None
    smoother = HomographySmoother(config)

    first_matrix, first_status = smoother.update(estimate)
    held_matrix, held_status = smoother.update(None)
    second_held_matrix, second_held_status = smoother.update(None)
    expired_matrix, expired_status = smoother.update(None)

    assert first_matrix is not None and first_status is MappingStatus.FRESH
    assert held_matrix is not None and held_status is MappingStatus.HELD
    assert second_held_matrix is not None
    assert second_held_status is MappingStatus.HELD
    assert expired_matrix is None
    assert expired_status is MappingStatus.UNAVAILABLE
