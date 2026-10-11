"""Detection, tracking, team/pitch analysis, annotation, and video output."""

from __future__ import annotations

import argparse
import logging
import subprocess
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv
from imageio_ffmpeg import get_ffmpeg_exe

from src.config import DEFAULT_CONFIG, AppConfig, OUTPUTS_DIR
from src.detect import FootballDetector
from src.pitch import MappingStatus, PitchMapper, PitchMapping, project_points
from src.teams import TeamClassifier, TeamPrediction, TeamRole
from src.viz import annotate_pitch_keypoints, overlay_radar, render_radar


LOGGER = logging.getLogger(__name__)
Frame = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Validated video metadata needed by the processing loop."""

    width: int
    height: int
    fps: float
    total_frames: int


@dataclass(frozen=True, slots=True)
class ProcessingReport:
    """Summary returned after a video has been processed."""

    output_path: Path
    frames_processed: int
    frames_with_ball: int
    elapsed_seconds: float
    team_model_ready: bool = False
    classified_tracks: int = 0
    frames_with_pitch_mapping: int = 0
    frames_with_fresh_pitch_mapping: int = 0


class H264VideoWriter:
    """Pipe raw frames to an isolated FFmpeg H.264 encoder process."""

    def __init__(self, path: Path, video_info: VideoInfo, config: AppConfig) -> None:
        self.path = path
        self._closed = False
        frame_size = f"{video_info.width}x{video_info.height}"
        command = [
            get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            frame_size,
            "-r",
            f"{video_info.fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            config.video.codec,
            "-preset",
            config.video.preset,
            "-crf",
            str(config.video.crf),
            "-pix_fmt",
            config.video.pixel_format,
            "-movflags",
            "+faststart",
            str(path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not create H.264 output {path}: {exc}") from exc
        self._width = video_info.width
        self._height = video_info.height

    def write(self, frame: Frame) -> None:
        """Encode one BGR frame."""

        if self._closed:
            raise RuntimeError("Cannot write to a closed video writer.")
        expected_shape = (self._height, self._width, 3)
        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame has shape {frame.shape}; expected {expected_shape}."
            )
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is unavailable.")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            error = self._read_error()
            raise RuntimeError(f"FFmpeg stopped while encoding: {error}") from exc

    def release(self) -> None:
        """Flush delayed frames and finalize the MP4 container."""

        if self._closed:
            return
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait()
        error = self._read_error()
        self._closed = True
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed to finalize {self.path} (exit {return_code}): {error}"
            )

    def _read_error(self) -> str:
        if self._process.stderr is None:
            return "no diagnostic output"
        message = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        return message or "no diagnostic output"


def read_video_info(capture: cv2.VideoCapture, fallback_fps: float) -> VideoInfo:
    """Read metadata and substitute a safe FPS when a container omits it."""

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise ValueError("The input video reports an invalid frame size.")
    if not np.isfinite(fps) or fps <= 0:
        LOGGER.warning(
            "Input video has no valid FPS; using %.3f FPS.",
            fallback_fps,
        )
        fps = fallback_fps
    return VideoInfo(width, height, fps, max(total_frames, 0))


def default_output_path(
    input_path: Path,
    *,
    classify_teams: bool = False,
    render_pitch_radar: bool = False,
) -> Path:
    """Build a non-destructive default output path under ``data/outputs``."""

    suffix = (
        "radar"
        if render_pitch_radar
        else ("teams" if classify_teams else "tracked")
    )
    return OUTPUTS_DIR / f"{input_path.stem}_{suffix}.mp4"


def _temporary_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def make_byte_tracker(config: AppConfig, frame_rate: float) -> sv.ByteTrack:
    """Build the configured ByteTrack instance for any video workflow."""

    settings = config.tracker
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The `ByteTrack` was deprecated.*",
            category=FutureWarning,
        )
        return sv.ByteTrack(
            track_activation_threshold=settings.track_activation_threshold,
            lost_track_buffer=settings.lost_track_buffer,
            minimum_matching_threshold=settings.minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=settings.minimum_consecutive_frames,
        )


class VideoTracker:
    """Stateful per-video detection, tracking, and annotation pipeline."""

    def __init__(
        self,
        config: AppConfig,
        *,
        classify_teams: bool = False,
        render_pitch_radar: bool = False,
        show_pitch_keypoints: bool = False,
    ) -> None:
        self.config = config
        self.render_pitch_radar = render_pitch_radar
        self.show_pitch_keypoints = show_pitch_keypoints
        self.classify_teams = classify_teams or render_pitch_radar
        self.detector = FootballDetector(config.detector)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.45,
        )
        self.team_classifier = (
            TeamClassifier(config.teams) if self.classify_teams else None
        )
        self.pitch_mapper = PitchMapper(config.pitch) if render_pitch_radar else None
        role_colors = {
            TeamRole.TEAM_A: "#2563EB",
            TeamRole.TEAM_B: "#DC2626",
            TeamRole.REFEREE: "#F59E0B",
            TeamRole.GOALKEEPER: "#A855F7",
            TeamRole.UNKNOWN: "#94A3B8",
            TeamRole.BALL: "#FACC15",
        }
        self.role_box_annotators = {
            role: sv.BoxAnnotator(color=sv.Color.from_hex(color), thickness=2)
            for role, color in role_colors.items()
        }
        self.role_label_annotators = {
            role: sv.LabelAnnotator(
                color=sv.Color.from_hex(color),
                text_thickness=1,
                text_scale=0.45,
            )
            for role, color in role_colors.items()
        }

    def process(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_frame: int = 0,
        max_frames: int | None = None,
        overwrite: bool = False,
    ) -> ProcessingReport:
        """Process one clip and atomically publish the completed output."""

        input_path = input_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        if input_path == output_path:
            raise ValueError("Input and output paths must be different.")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --overwrite to replace it."
            )
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative.")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = _temporary_output_path(output_path)
        if temporary_path.exists():
            raise FileExistsError(
                f"A partial output already exists: {temporary_path}. "
                "Move or remove it before retrying."
            )

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open input video: {input_path}")

        writer: H264VideoWriter | None = None
        started_at = perf_counter()
        frames_processed = 0
        frames_with_ball = 0
        frames_with_pitch_mapping = 0
        frames_with_fresh_pitch_mapping = 0
        completed = False

        try:
            video_info = read_video_info(capture, self.config.video.fallback_fps)
            if video_info.total_frames and start_frame >= video_info.total_frames:
                raise ValueError(
                    f"start_frame {start_frame} is outside a "
                    f"{video_info.total_frames}-frame video."
                )
            if start_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if abs(actual_frame - start_frame) > 1:
                    LOGGER.warning(
                        "Video seek requested frame %d but reached frame %d.",
                        start_frame,
                        actual_frame,
                    )
            tracker = self._make_tracker(video_info.fps)
            if self.team_classifier is not None:
                self.team_classifier.reset()
            if self.pitch_mapper is not None:
                self.pitch_mapper.reset()
            writer = self._open_writer(temporary_path, video_info)

            while max_frames is None or frames_processed < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = self.detector.detect(frame)
                tracked = tracker.update_with_detections(detections)
                if self._contains_ball(tracked):
                    frames_with_ball += 1

                team_predictions = self._classify_teams(
                    frame,
                    tracked,
                    frames_processed,
                )
                pitch_mapping = (
                    self.pitch_mapper.map_frame(frame)
                    if self.pitch_mapper is not None
                    else None
                )
                if pitch_mapping is not None and pitch_mapping.is_available:
                    frames_with_pitch_mapping += 1
                    if pitch_mapping.is_fresh:
                        frames_with_fresh_pitch_mapping += 1
                annotated = self._annotate(frame, tracked, team_predictions)
                if pitch_mapping is not None:
                    annotated = self._annotate_pitch_mapping(
                        annotated,
                        tracked,
                        team_predictions,
                        pitch_mapping,
                    )
                writer.write(annotated)
                frames_processed += 1

                interval = self.config.video.progress_interval_frames
                if interval > 0 and frames_processed % interval == 0:
                    LOGGER.info(
                        "Processed %d%s frames",
                        frames_processed,
                        f"/{video_info.total_frames}" if video_info.total_frames else "",
                    )

            if frames_processed == 0:
                raise RuntimeError("No frames could be decoded from the input video.")
            completed = True
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if not completed and temporary_path.exists():
                temporary_path.unlink()

        # ``replace`` is atomic on the same filesystem. Existing files are only
        # replaced after an explicit --overwrite request checked above.
        temporary_path.replace(output_path)
        return ProcessingReport(
            output_path=output_path,
            frames_processed=frames_processed,
            frames_with_ball=frames_with_ball,
            elapsed_seconds=perf_counter() - started_at,
            team_model_ready=(
                self.team_classifier.is_ready
                if self.team_classifier is not None
                else False
            ),
            classified_tracks=(
                self.team_classifier.classified_track_count
                if self.team_classifier is not None
                else 0
            ),
            frames_with_pitch_mapping=frames_with_pitch_mapping,
            frames_with_fresh_pitch_mapping=frames_with_fresh_pitch_mapping,
        )

    def _make_tracker(self, frame_rate: float) -> sv.ByteTrack:
        # Supervision 0.30 still bundles ByteTrack, as requested for this
        # project, but announces its move to Roboflow's separate `trackers`
        # package. requirements.txt pins <0.31 so this API cannot disappear
        # underneath us; a controlled migration can happen in a later stage.
        return make_byte_tracker(self.config, frame_rate)

    def _open_writer(
        self,
        path: Path,
        video_info: VideoInfo,
    ) -> H264VideoWriter:
        return H264VideoWriter(path, video_info, self.config)

    def _contains_ball(self, detections: sv.Detections) -> bool:
        if detections.class_id is None:
            return False
        return any(
            self.detector.canonical_label(int(class_id)) == "ball"
            for class_id in detections.class_id
        )

    def _classify_teams(
        self,
        frame: Frame,
        detections: sv.Detections,
        frame_index: int,
    ) -> list[TeamPrediction] | None:
        if self.team_classifier is None:
            return None
        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else np.full(len(detections), -1, dtype=int)
        )
        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else np.full(len(detections), -1, dtype=int)
        )
        labels = [
            self.detector.canonical_label(int(class_id)) for class_id in class_ids
        ]
        return self.team_classifier.update(
            frame,
            detections.xyxy,
            tracker_ids,
            labels,
            frame_index,
        )

    def _annotate(
        self,
        frame: Frame,
        detections: sv.Detections,
        team_predictions: list[TeamPrediction] | None = None,
    ) -> Frame:
        labels = self._labels(detections, team_predictions)
        if team_predictions is None:
            annotated = self.box_annotator.annotate(
                scene=frame.copy(),
                detections=detections,
            )
            annotated = self.label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels,
            )
        else:
            annotated = frame.copy()
            for role in TeamRole:
                mask = np.array(
                    [prediction.role is role for prediction in team_predictions],
                    dtype=bool,
                )
                if not np.any(mask):
                    continue
                role_detections = detections[mask]
                role_labels = [
                    label for label, include in zip(labels, mask, strict=True) if include
                ]
                annotated = self.role_box_annotators[role].annotate(
                    scene=annotated,
                    detections=role_detections,
                )
                annotated = self.role_label_annotators[role].annotate(
                    scene=annotated,
                    detections=role_detections,
                    labels=role_labels,
                )

        # A broadcast ball box can be only a few pixels wide. Add a ring so it
        # remains visible without replacing Supervision's regular annotation.
        if detections.class_id is not None:
            for box, class_id in zip(
                detections.xyxy,
                detections.class_id,
                strict=True,
            ):
                if self.detector.canonical_label(int(class_id)) != "ball":
                    continue
                center = (
                    int(round((box[0] + box[2]) / 2)),
                    int(round((box[1] + box[3]) / 2)),
                )
                cv2.circle(annotated, center, 10, (0, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def _annotate_pitch_mapping(
        self,
        frame: Frame,
        detections: sv.Detections,
        team_predictions: list[TeamPrediction] | None,
        mapping: PitchMapping,
    ) -> Frame:
        """Project tracked ground anchors and overlay a top-down radar."""

        annotated = frame
        if self.show_pitch_keypoints:
            annotated = annotate_pitch_keypoints(
                annotated,
                mapping.keypoints,
                self.config.pitch.keypoint_confidence,
            )

        roles: list[TeamRole] = []
        tracker_ids: list[int] = []
        image_anchors: list[tuple[float, float]] = []
        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else np.full(len(detections), -1, dtype=int)
        )
        detection_tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else np.full(len(detections), -1, dtype=int)
        )
        for index, (box, class_id, tracker_id) in enumerate(
            zip(
                detections.xyxy,
                class_ids,
                detection_tracker_ids,
                strict=True,
            )
        ):
            label = self.detector.canonical_label(int(class_id))
            # A bottom-centre point is the best available ground contact for a
            # box. An airborne ball cannot be localized correctly from one view.
            image_anchors.append(
                (float((box[0] + box[2]) / 2), float(box[3]))
            )
            tracker_ids.append(int(tracker_id))
            if team_predictions is not None:
                roles.append(team_predictions[index].role)
            elif label == "ball":
                roles.append(TeamRole.BALL)
            else:
                roles.append(TeamRole.UNKNOWN)

        pitch_points = np.empty((0, 2), dtype=np.float64)
        radar_status = mapping.status
        if mapping.homography is not None and image_anchors:
            try:
                pitch_points = project_points(image_anchors, mapping.homography)
            except ValueError:
                roles.clear()
                tracker_ids.clear()
                radar_status = MappingStatus.UNAVAILABLE
        elif mapping.homography is None:
            roles.clear()
            tracker_ids.clear()

        radar = render_radar(
            pitch_points,
            roles,
            tracker_ids,
            radar_status,
            self.config.pitch,
            self.pitch_mapper.pitch if self.pitch_mapper is not None else None,
        )
        return overlay_radar(annotated, radar, self.config.pitch)

    def _labels(
        self,
        detections: sv.Detections,
        team_predictions: list[TeamPrediction] | None = None,
    ) -> list[str]:
        labels: list[str] = []
        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else np.full(len(detections), -1, dtype=int)
        )
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.full(len(detections), np.nan, dtype=float)
        )
        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else np.full(len(detections), -1, dtype=int)
        )

        if team_predictions is not None and len(team_predictions) != len(detections):
            raise ValueError("Expected one team prediction per detection.")

        values = zip(class_ids, confidences, tracker_ids, strict=True)
        for index, (class_id, confidence, tracker_id) in enumerate(values):
            name = (
                team_predictions[index].role.value
                if team_predictions is not None
                else self.detector.canonical_label(int(class_id))
            )
            track_text = f"#{int(tracker_id)}" if int(tracker_id) >= 0 else "#?"
            confidence_text = (
                f" {float(confidence):.2f}" if np.isfinite(confidence) else ""
            )
            labels.append(f"{name} {track_text}{confidence_text}")
        return labels


def build_parser() -> argparse.ArgumentParser:
    """Create the detection/tracking/team-split command-line parser."""

    defaults = DEFAULT_CONFIG.detector
    parser = argparse.ArgumentParser(
        description="Detect and ByteTrack people/ball in a football clip.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output MP4 path (default suffix: _tracked, _teams, or _radar)."
        ),
    )
    parser.add_argument(
        "--teams",
        action="store_true",
        help="Enable Stage 2 jersey-color Team A/B classification.",
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help="Enable Stage 3 pitch mapping and top-down radar (also enables teams).",
    )
    parser.add_argument(
        "--pitch-model",
        default=DEFAULT_CONFIG.pitch.model_path,
        help="Local 32-keypoint football-pitch .pt model.",
    )
    parser.add_argument(
        "--pitch-device",
        default=DEFAULT_CONFIG.pitch.device,
        help="Pitch model device; defaults to the value of --device.",
    )
    parser.add_argument(
        "--pitch-imgsz",
        type=int,
        default=DEFAULT_CONFIG.pitch.image_size,
        help="Pitch-keypoint inference image size.",
    )
    parser.add_argument(
        "--pitch-conf",
        type=float,
        default=DEFAULT_CONFIG.pitch.detection_confidence,
        help="Whole-pitch detection confidence threshold.",
    )
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=DEFAULT_CONFIG.pitch.keypoint_confidence,
        help="Per-landmark confidence threshold for homography fitting.",
    )
    parser.add_argument(
        "--show-pitch-keypoints",
        action="store_true",
        help="Draw numbered calibration landmarks on the video for debugging.",
    )
    parser.add_argument(
        "--model",
        default=defaults.model_path,
        help="Ultralytics model name or local .pt path.",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        help="Ultralytics device, e.g. cpu, mps, 0, or 0,1.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=defaults.image_size,
        help="YOLO inference image size.",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=defaults.person_confidence,
        help="Person/player confidence threshold.",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=defaults.ball_confidence,
        help="Ball confidence threshold.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start processing at this zero-based source frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after processing succeeds.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Apply explicit CLI overrides to the centralized defaults."""

    detector = replace(
        DEFAULT_CONFIG.detector,
        model_path=args.model,
        device=args.device,
        image_size=args.imgsz,
        person_confidence=args.person_conf,
        ball_confidence=args.ball_conf,
    )
    pitch = replace(
        DEFAULT_CONFIG.pitch,
        model_path=args.pitch_model,
        device=args.pitch_device or args.device,
        image_size=args.pitch_imgsz,
        detection_confidence=args.pitch_conf,
        keypoint_confidence=args.keypoint_conf,
    )
    if detector.image_size <= 0:
        raise ValueError("--imgsz must be positive.")
    if pitch.image_size <= 0:
        raise ValueError("--pitch-imgsz must be positive.")
    for name, value in (
        ("--person-conf", detector.person_confidence),
        ("--ball-conf", detector.ball_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    for name, value in (
        ("--pitch-conf", pitch.detection_confidence),
        ("--keypoint-conf", pitch.keypoint_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    return replace(DEFAULT_CONFIG, detector=detector, pitch=pitch)


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    try:
        config = config_from_args(args)
        output_path = args.output or default_output_path(
            args.input,
            classify_teams=args.teams,
            render_pitch_radar=args.radar,
        )
        report = VideoTracker(
            config,
            classify_teams=args.teams,
            render_pitch_radar=args.radar,
            show_pitch_keypoints=args.show_pitch_keypoints,
        ).process(
            args.input,
            output_path,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Saved %d frames to %s in %.1fs (ball present in %d frames).",
        report.frames_processed,
        report.output_path,
        report.elapsed_seconds,
        report.frames_with_ball,
    )
    if args.teams or args.radar:
        if report.team_model_ready:
            LOGGER.info(
                "Team model ready; %d sampled tracks received stable team labels.",
                report.classified_tracks,
            )
        else:
            LOGGER.warning(
                "Team model never became ready; use a clip with at least %d "
                "well-visible players from both teams on the pitch.",
                config.teams.min_tracks_to_fit,
            )
    if args.radar:
        LOGGER.info(
            "Pitch mapping available in %d/%d frames (%d fresh estimates).",
            report.frames_with_pitch_mapping,
            report.frames_processed,
            report.frames_with_fresh_pitch_mapping,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
