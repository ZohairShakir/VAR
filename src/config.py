"""Central configuration for the mini VAR pipeline.

Stage-specific modules should take these immutable dataclasses as inputs instead
of scattering thresholds and paths throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
WEIGHTS_DIR = DATA_DIR / "weights"
OUTPUTS_DIR = DATA_DIR / "outputs"
UPLOADS_DIR = DATA_DIR / "uploads"


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """YOLO inference settings.

    Ultralytics downloads the default weights on first use. Football-specific
    weights should also live under ``data/weights``.
    """

    model_path: str = str(WEIGHTS_DIR / "yolo11n.pt")
    image_size: int = 1280
    person_confidence: float = 0.25
    ball_confidence: float = 0.10
    iou_threshold: float = 0.50
    device: str | None = None

    # COCO names plus common names used by football-specific detectors.
    target_class_names: tuple[str, ...] = (
        "person",
        "sports ball",
        "player",
        "goalkeeper",
        "referee",
        "ball",
    )


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Supervision ByteTrack settings."""

    track_activation_threshold: float = 0.15
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.80
    minimum_consecutive_frames: int = 1


@dataclass(frozen=True, slots=True)
class TeamConfig:
    """Online jersey-color team classification settings."""

    sample_every_n_frames: int = 5
    max_samples_per_track: int = 25
    min_samples_per_track: int = 2
    min_tracks_to_fit: int = 6
    min_tracks_per_cluster: int = 2
    refit_every_new_samples: int = 40
    random_state: int = 17

    # Relative torso crop inside a person bounding box.
    torso_left_ratio: float = 0.20
    torso_right_ratio: float = 0.80
    torso_top_ratio: float = 0.18
    torso_bottom_ratio: float = 0.62
    min_person_box_height: int = 36
    min_valid_torso_pixels: int = 64

    # OpenCV HSV thresholds used to reject grass pixels.
    green_hue_min: int = 30
    green_hue_max: int = 95
    green_saturation_min: int = 45
    green_value_min: int = 35
    min_pitch_green_fraction: float = 0.10
    pitch_sample_stride: int = 8

    # Confidence gates in OpenCV's 8-bit CIELAB coordinate system.
    min_cluster_separation: float = 18.0
    min_cluster_radius: float = 20.0
    cluster_radius_mad_scale: float = 3.0
    max_center_distance: float = 55.0
    min_assignment_margin: float = 0.12


@dataclass(frozen=True, slots=True)
class PitchConfig:
    """Pitch-keypoint inference, homography, and radar settings."""

    model_path: str = str(WEIGHTS_DIR / "football-pitch-detection.pt")
    image_size: int = 640
    detection_confidence: float = 0.25
    keypoint_confidence: float = 0.50
    device: str | None = None

    # RANSAC operates in pitch coordinates, so these thresholds are metres.
    min_visible_keypoints: int = 5
    min_inliers: int = 4
    min_inlier_ratio: float = 0.55
    ransac_reprojection_threshold_m: float = 1.5
    max_median_reprojection_error_m: float = 0.75

    # Homographies are smoothed only while the preceding transform still
    # explains the current keypoints. A larger shift is treated as a pan/cut.
    smoothing_alpha: float = 0.40
    max_previous_error_m: float = 8.0
    max_smoothed_error_m: float = 2.0
    max_stale_frames: int = 5

    radar_width: int = 360
    radar_margin: int = 12
    radar_opacity: float = 0.90


@dataclass(frozen=True, slots=True)
class PoseConfig:
    """Human-pose inference settings for a selected kick frame."""

    model_path: str = str(WEIGHTS_DIR / "yolo11n-pose.pt")
    image_size: int = 640
    detection_confidence: float = 0.15
    keypoint_confidence: float = 0.25
    crop_padding_ratio: float = 0.10
    min_legal_keypoints: int = 3
    device: str | None = None


@dataclass(frozen=True, slots=True)
class OffsideConfig:
    """Stage 4 frame context and decision thresholds."""

    warmup_frames: int = 150
    decision_uncertainty_m: float = 0.50
    pitch_bounds_margin_m: float = 2.0


@dataclass(frozen=True, slots=True)
class VideoConfig:
    """Video encoding settings."""

    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    crf: int = 23
    preset: str = "fast"
    fallback_fps: float = 30.0
    progress_interval_frames: int = 100


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete application configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    offside: OffsideConfig = field(default_factory=OffsideConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


DEFAULT_CONFIG = AppConfig()
