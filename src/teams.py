"""Stage 2 jersey-color extraction and temporally stable team clustering."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from src.config import TeamConfig


LOGGER = logging.getLogger(__name__)
Frame = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]


class TeamRole(str, Enum):
    """Stable labels consumed by annotation and later offside logic."""

    TEAM_A = "Team A"
    TEAM_B = "Team B"
    REFEREE = "Referee"
    GOALKEEPER = "Goalkeeper"
    UNKNOWN = "Unknown"
    BALL = "Ball"


@dataclass(frozen=True, slots=True)
class TeamPrediction:
    """Team/role prediction for one tracked detection."""

    role: TeamRole
    confidence: float
    center_distance: float | None = None


def _green_mask(hsv: npt.NDArray[np.uint8], config: TeamConfig) -> npt.NDArray[np.bool_]:
    """Return pixels likely to be grass in an OpenCV HSV image."""

    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return (
        (hue >= config.green_hue_min)
        & (hue <= config.green_hue_max)
        & (saturation >= config.green_saturation_min)
        & (value >= config.green_value_min)
    )


def pitch_green_fraction(frame: Frame, config: TeamConfig) -> float:
    """Estimate how much of a frame is pitch using a cheap strided sample."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR frame with shape (height, width, 3).")
    stride = max(config.pitch_sample_stride, 1)
    sample = frame[::stride, ::stride]
    if sample.size == 0:
        return 0.0
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    return float(np.mean(_green_mask(hsv, config)))


def extract_jersey_feature(
    frame: Frame,
    box_xyxy: npt.ArrayLike,
    config: TeamConfig,
) -> FloatArray | None:
    """Extract a robust median CIELAB color from a player's torso.

    The central upper-body crop reduces contamination from faces, arms, shorts,
    neighboring players, and pitch. Remaining grass-colored pixels are removed
    before taking the median.
    """

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR frame with shape (height, width, 3).")

    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height < config.min_person_box_height:
        return None

    torso_x1 = int(np.floor(x1 + config.torso_left_ratio * width))
    torso_x2 = int(np.ceil(x1 + config.torso_right_ratio * width))
    torso_y1 = int(np.floor(y1 + config.torso_top_ratio * height))
    torso_y2 = int(np.ceil(y1 + config.torso_bottom_ratio * height))

    torso_x1 = int(np.clip(torso_x1, 0, frame_width))
    torso_x2 = int(np.clip(torso_x2, 0, frame_width))
    torso_y1 = int(np.clip(torso_y1, 0, frame_height))
    torso_y2 = int(np.clip(torso_y2, 0, frame_height))
    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None

    torso = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    jersey_pixels = lab[~_green_mask(hsv, config)]
    if len(jersey_pixels) < config.min_valid_torso_pixels:
        return None
    return np.median(jersey_pixels, axis=0).astype(np.float64)


class TeamClassifier:
    """Accumulate per-track jersey colors and split them into two teams.

    Each track contributes one median prototype when K-means is fitted, so a
    player visible for a long time cannot dominate the centers. Cluster indices
    are aligned after every refit to keep Team A/B stable over time.
    """

    def __init__(self, config: TeamConfig) -> None:
        self.config = config
        self._samples_by_track: dict[int, deque[FloatArray]] = defaultdict(
            lambda: deque(maxlen=config.max_samples_per_track)
        )
        self._centers: FloatArray | None = None
        self._center_radii: FloatArray | None = None
        self._new_samples_since_fit = 0
        self._fit_track_ids: set[int] = set()

    @property
    def is_ready(self) -> bool:
        """Whether two sufficiently separated team centers are available."""

        return self._centers is not None

    @property
    def centers(self) -> FloatArray | None:
        """Return a defensive copy of the current Team A/B centers."""

        return None if self._centers is None else self._centers.copy()

    @property
    def classified_track_count(self) -> int:
        """Number of sampled tracks currently assignable to a team."""

        return sum(
            self._prediction_for_track(track_id).role
            in {TeamRole.TEAM_A, TeamRole.TEAM_B}
            for track_id in self._samples_by_track
        )

    def reset(self) -> None:
        """Clear all state before processing a different video."""

        self._samples_by_track.clear()
        self._centers = None
        self._center_radii = None
        self._new_samples_since_fit = 0
        self._fit_track_ids.clear()

    def update(
        self,
        frame: Frame,
        boxes_xyxy: npt.ArrayLike,
        tracker_ids: npt.ArrayLike,
        object_labels: list[str],
        frame_index: int,
    ) -> list[TeamPrediction]:
        """Observe one frame and return a role prediction per detection."""

        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
        ids = np.asarray(tracker_ids, dtype=int)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes_xyxy must have shape (N, 4).")
        if len(boxes) != len(ids) or len(boxes) != len(object_labels):
            raise ValueError("Boxes, tracker IDs, and labels must have equal lengths.")

        should_sample = (
            frame_index % max(self.config.sample_every_n_frames, 1) == 0
            and pitch_green_fraction(frame, self.config)
            >= self.config.min_pitch_green_fraction
        )
        if should_sample:
            for box, tracker_id, label in zip(
                boxes,
                ids,
                object_labels,
                strict=True,
            ):
                if tracker_id < 0 or not self._is_team_candidate(label):
                    continue
                feature = extract_jersey_feature(frame, box, self.config)
                if feature is None:
                    continue
                self._samples_by_track[int(tracker_id)].append(feature)
                self._new_samples_since_fit += 1

        self._fit_if_needed()
        return [
            self._prediction_for_detection(int(track_id), label)
            for track_id, label in zip(ids, object_labels, strict=True)
        ]

    @staticmethod
    def _is_team_candidate(label: str) -> bool:
        return label.lower() in {"person", "player"}

    def _track_prototype(self, track_id: int) -> FloatArray | None:
        samples = self._samples_by_track.get(track_id)
        if samples is None or len(samples) < self.config.min_samples_per_track:
            return None
        return np.median(np.stack(samples), axis=0)

    def _eligible_prototypes(self) -> tuple[list[int], FloatArray] | None:
        track_ids: list[int] = []
        prototypes: list[FloatArray] = []
        for track_id in sorted(self._samples_by_track):
            prototype = self._track_prototype(track_id)
            if prototype is not None:
                track_ids.append(track_id)
                prototypes.append(prototype)
        if len(prototypes) < self.config.min_tracks_to_fit:
            return None
        return track_ids, np.stack(prototypes)

    def _fit_if_needed(self) -> None:
        eligible = self._eligible_prototypes()
        if eligible is None:
            return
        track_ids, features = eligible
        eligible_ids = set(track_ids)
        if (
            self._centers is not None
            and self._new_samples_since_fit < self.config.refit_every_new_samples
        ):
            return

        # Some restricted/macOS environments do not expose physical-core
        # metadata to joblib. K-means still runs correctly on logical cores.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Could not find the number of physical cores.*",
                category=UserWarning,
            )
            model = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=10,
                random_state=self.config.random_state,
            ).fit(features)
        counts = np.bincount(model.labels_, minlength=2)
        if int(np.min(counts)) < self.config.min_tracks_per_cluster:
            return

        centers = model.cluster_centers_.astype(np.float64)
        separation = float(np.linalg.norm(centers[0] - centers[1]))
        if separation < self.config.min_cluster_separation:
            return

        centers = self._align_centers(centers)
        center_radii = self._estimate_center_radii(features, centers)
        first_fit = self._centers is None
        self._centers = centers
        self._center_radii = center_radii
        self._new_samples_since_fit = 0
        self._fit_track_ids = eligible_ids
        log = LOGGER.info if first_fit else LOGGER.debug
        log(
            "Team color model fitted with %d tracks (center separation %.1f).",
            len(track_ids),
            separation,
        )

    def _align_centers(self, centers: FloatArray) -> FloatArray:
        """Keep cluster index 0/1 stable across K-means refits."""

        if self._centers is None:
            # Deterministic initial naming. Team A/B are still arbitrary labels.
            order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
            return centers[order]

        direct_cost = float(
            np.linalg.norm(centers[0] - self._centers[0])
            + np.linalg.norm(centers[1] - self._centers[1])
        )
        swapped_cost = float(
            np.linalg.norm(centers[1] - self._centers[0])
            + np.linalg.norm(centers[0] - self._centers[1])
        )
        return centers if direct_cost <= swapped_cost else centers[::-1]

    def _estimate_center_radii(
        self,
        features: FloatArray,
        centers: FloatArray,
    ) -> FloatArray:
        """Estimate a robust acceptance radius around each team center.

        A median/MAD radius keeps a lone goalkeeper or referee from expanding a
        team's accepted color range. The global maximum remains a hard ceiling.
        """

        all_distances = np.linalg.norm(
            features[:, None, :] - centers[None, :, :],
            axis=2,
        )
        assignments = np.argmin(all_distances, axis=1)
        radii: list[float] = []
        for center_index in range(2):
            distances = all_distances[assignments == center_index, center_index]
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median)))
            robust_radius = median + self.config.cluster_radius_mad_scale * max(mad, 1.0)
            radii.append(
                float(
                    np.clip(
                        robust_radius,
                        self.config.min_cluster_radius,
                        self.config.max_center_distance,
                    )
                )
            )
        return np.asarray(radii, dtype=np.float64)

    def _prediction_for_detection(
        self,
        track_id: int,
        label: str,
    ) -> TeamPrediction:
        normalized = label.lower()
        if normalized == "ball":
            return TeamPrediction(TeamRole.BALL, 1.0)
        if normalized == "referee":
            return TeamPrediction(TeamRole.REFEREE, 1.0)
        if normalized == "goalkeeper":
            return TeamPrediction(TeamRole.GOALKEEPER, 1.0)
        if track_id < 0 or not self._is_team_candidate(normalized):
            return TeamPrediction(TeamRole.UNKNOWN, 0.0)
        return self._prediction_for_track(track_id)

    def _prediction_for_track(self, track_id: int) -> TeamPrediction:
        prototype = self._track_prototype(track_id)
        if prototype is None or self._centers is None:
            return TeamPrediction(TeamRole.UNKNOWN, 0.0)

        distances = np.linalg.norm(self._centers - prototype, axis=1)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        other_distance = float(distances[1 - best_index])
        acceptance_radius = (
            float(self._center_radii[best_index])
            if self._center_radii is not None
            else self.config.max_center_distance
        )
        margin = (other_distance - best_distance) / max(other_distance, 1e-9)
        distance_score = np.clip(
            1.0 - best_distance / acceptance_radius,
            0.0,
            1.0,
        )
        confidence = float(np.sqrt(max(margin, 0.0) * distance_score))
        if (
            best_distance > acceptance_radius
            or margin < self.config.min_assignment_margin
        ):
            return TeamPrediction(TeamRole.UNKNOWN, confidence, best_distance)

        role = TeamRole.TEAM_A if best_index == 0 else TeamRole.TEAM_B
        return TeamPrediction(role, confidence, best_distance)
