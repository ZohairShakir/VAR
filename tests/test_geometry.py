"""Geometry regression tests shared by the later pitch/offside stages."""

from __future__ import annotations

import numpy as np
import pytest

from src.pitch import project_points


def test_project_points_with_synthetic_homography() -> None:
    """A known projective transform should produce known pitch coordinates."""

    image_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ]
    )
    # Includes scale, translation, and a small perspective component.
    homography = np.array(
        [
            [2.0, 0.0, 5.0],
            [0.0, 3.0, 7.0],
            [0.01, 0.02, 1.0],
        ]
    )
    expected = np.array(
        [
            [5.0, 7.0],
            [25.0 / 1.1, 7.0 / 1.1],
            [5.0 / 1.2, 37.0 / 1.2],
            [25.0 / 1.3, 37.0 / 1.3],
        ]
    )

    np.testing.assert_allclose(
        project_points(image_points, homography),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_project_points_rejects_point_at_infinity() -> None:
    homography = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="maps to infinity"):
        project_points([[0.0, 5.0]], homography)

