"""Tests for torso color extraction and stable two-team clustering."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from src.config import TeamConfig
from src.teams import (
    TeamClassifier,
    TeamRole,
    extract_jersey_feature,
    pitch_green_fraction,
)


def _team_frame(
    colors: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((140, 240, 3), (0, 150, 0), dtype=np.uint8)
    boxes: list[list[float]] = []
    for index, color in enumerate(colors):
        x1 = 10 + index * 36
        x2 = x1 + 24
        y1, y2 = 20, 120
        frame[y1:y2, x1:x2] = color
        boxes.append([x1, y1, x2, y2])
    return frame, np.asarray(boxes, dtype=np.float64)


def test_extract_jersey_feature_uses_torso_and_rejects_grass() -> None:
    config = replace(TeamConfig(), min_person_box_height=20)
    frame = np.full((100, 80, 3), (0, 160, 0), dtype=np.uint8)
    frame[10:90, 20:60] = (0, 0, 255)

    feature = extract_jersey_feature(frame, [20, 10, 60, 90], config)
    expected = cv2.cvtColor(
        np.array([[[0, 0, 255]]], dtype=np.uint8),
        cv2.COLOR_BGR2LAB,
    )[0, 0].astype(np.float64)

    assert feature is not None
    np.testing.assert_allclose(feature, expected, atol=1.0)


def test_pitch_green_fraction_distinguishes_pitch_from_graphic() -> None:
    config = TeamConfig()
    pitch = np.full((100, 100, 3), (0, 150, 0), dtype=np.uint8)
    graphic = np.full((100, 100, 3), (180, 20, 180), dtype=np.uint8)

    assert pitch_green_fraction(pitch, config) > 0.95
    assert pitch_green_fraction(graphic, config) < 0.05


def test_team_classifier_splits_colors_and_keeps_roles_stable() -> None:
    config = replace(
        TeamConfig(),
        sample_every_n_frames=1,
        min_samples_per_track=2,
        min_tracks_to_fit=6,
        refit_every_new_samples=1,
        min_pitch_green_fraction=0.05,
    )
    classifier = TeamClassifier(config)
    colors = [(255, 0, 0)] * 3 + [(0, 0, 255)] * 3
    frame, boxes = _team_frame(colors)
    ids = np.arange(1, 7)
    labels = ["person"] * 6

    first = classifier.update(frame, boxes, ids, labels, frame_index=0)
    assert all(prediction.role is TeamRole.UNKNOWN for prediction in first)

    second = classifier.update(frame, boxes, ids, labels, frame_index=1)
    assert classifier.is_ready
    left_role = second[0].role
    right_role = second[3].role
    assert left_role in {TeamRole.TEAM_A, TeamRole.TEAM_B}
    assert right_role in {TeamRole.TEAM_A, TeamRole.TEAM_B}
    assert left_role is not right_role
    assert all(prediction.role is left_role for prediction in second[:3])
    assert all(prediction.role is right_role for prediction in second[3:])

    noisy_colors = [(235, 15, 10)] * 3 + [(10, 15, 235)] * 3
    noisy_frame, _ = _team_frame(noisy_colors)
    third = classifier.update(noisy_frame, boxes, ids, labels, frame_index=2)
    assert all(prediction.role is left_role for prediction in third[:3])
    assert all(prediction.role is right_role for prediction in third[3:])


def test_explicit_roles_are_never_forced_into_team_clusters() -> None:
    classifier = TeamClassifier(TeamConfig())
    frame, boxes = _team_frame([(0, 0, 0)] * 3)
    predictions = classifier.update(
        frame,
        boxes,
        [10, 11, 12],
        ["referee", "goalkeeper", "ball"],
        frame_index=0,
    )

    assert [prediction.role for prediction in predictions] == [
        TeamRole.REFEREE,
        TeamRole.GOALKEEPER,
        TeamRole.BALL,
    ]


def test_color_outlier_is_left_unknown() -> None:
    config = replace(
        TeamConfig(),
        sample_every_n_frames=1,
        min_samples_per_track=2,
        min_tracks_to_fit=6,
        refit_every_new_samples=100,
        min_pitch_green_fraction=0.05,
    )
    classifier = TeamClassifier(config)
    team_frame, team_boxes = _team_frame(
        [(255, 0, 0)] * 3 + [(255, 255, 255)] * 3
    )
    team_ids = np.arange(1, 7)
    for frame_index in range(2):
        classifier.update(
            team_frame,
            team_boxes,
            team_ids,
            ["person"] * 6,
            frame_index,
        )
    assert classifier.is_ready

    outlier_frame, outlier_box = _team_frame([(0, 0, 255)])
    classifier.update(
        outlier_frame,
        outlier_box,
        [99],
        ["person"],
        frame_index=2,
    )
    prediction = classifier.update(
        outlier_frame,
        outlier_box,
        [99],
        ["person"],
        frame_index=3,
    )[0]

    assert prediction.role is TeamRole.UNKNOWN
