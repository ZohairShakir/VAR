"""Stage 4 single-frame pose projection and offside-position analysis."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from src.config import DEFAULT_CONFIG, AppConfig, OUTPUTS_DIR
from src.detect import FootballDetector
from src.pitch import (
    MappingStatus,
    PitchMapper,
    PitchMapping,
    SoccerPitch,
    project_points,
)
from src.pose import PlayerPose, PoseEstimator
from src.teams import TeamClassifier, TeamPrediction, TeamRole
from src.track import make_byte_tracker, read_video_info
from src.viz import (
    ROLE_COLORS,
    annotate_player_poses,
    draw_decision_banner,
    draw_pitch_x_line,
    overlay_radar,
    render_radar,
)


LOGGER = logging.getLogger(__name__)
Frame = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]


class AttackDirection(str, Enum):
    """Direction in which the attacking team approaches the goal."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def sign(self) -> float:
        return 1.0 if self is AttackDirection.RIGHT else -1.0


class DecisionStatus(str, Enum):
    """Stage 4 result states."""

    OFFSIDE = "offside"
    ONSIDE = "onside"
    UNCERTAIN = "uncertain"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ProjectedPlayer:
    """Offside-relevant image keypoints projected into pitch coordinates."""

    tracker_id: int
    role: TeamRole
    image_points: FloatArray
    pitch_points: FloatArray

    def __post_init__(self) -> None:
        if self.image_points.ndim != 2 or self.image_points.shape[1] != 2:
            raise ValueError("image_points must have shape (N, 2).")
        if self.pitch_points.shape != self.image_points.shape:
            raise ValueError("Image and pitch points must have matching shapes.")
        if len(self.pitch_points) == 0:
            raise ValueError("A projected player needs at least one legal keypoint.")

    def reference_index(self, direction: AttackDirection) -> int:
        """Return the legal body part nearest the attacked goal line."""

        values = direction.sign * self.pitch_points[:, 0]
        return int(np.argmax(values))

    def reference_x(self, direction: AttackDirection) -> float:
        return float(self.pitch_points[self.reference_index(direction), 0])

    def reference_image_point(self, direction: AttackDirection) -> FloatArray:
        return self.image_points[self.reference_index(direction)].copy()


@dataclass(frozen=True, slots=True)
class OffsideDecision:
    """Decision and geometry needed by the image and future UI layers."""

    status: DecisionStatus
    reason: str
    attacker_id: int | None = None
    second_last_opponent_id: int | None = None
    defender_line_x_m: float | None = None
    ball_x_m: float | None = None
    offside_line_x_m: float | None = None
    margin_m: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "attacker_id": self.attacker_id,
            "second_last_opponent_id": self.second_last_opponent_id,
            "defender_line_x_m": self.defender_line_x_m,
            "ball_x_m": self.ball_x_m,
            "offside_line_x_m": self.offside_line_x_m,
            "margin_m": self.margin_m,
        }


@dataclass(frozen=True, slots=True)
class KickFrameContext:
    """Detection, tracking, and team state at the chosen source frame."""

    frame: Frame
    detections: sv.Detections
    team_predictions: list[TeamPrediction]
    team_model_ready: bool


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    output_path: Path
    json_path: Path
    frame_index: int
    decision: OffsideDecision
    projected_player_count: int


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """In-memory decision and renderings shared by the CLI and Streamlit UI."""

    frame_index: int
    decision: OffsideDecision
    projected_player_count: int
    mapping_status: MappingStatus
    camera_view: Frame
    radar_view: Frame
    combined_view: Frame
    payload: dict[str, object]


def project_player_pose(
    pose: PlayerPose,
    role: TeamRole,
    homography: npt.ArrayLike,
    config: AppConfig,
    pitch: SoccerPitch | None = None,
) -> ProjectedPlayer | None:
    """Project confident legal body parts and reject impossible coordinates."""

    geometry = pitch or SoccerPitch()
    legal = pose.legal_mask(config.pose.keypoint_confidence)
    image_points = pose.keypoints_xy[legal]
    if len(image_points) < config.pose.min_legal_keypoints:
        return None
    try:
        pitch_points = project_points(image_points, homography)
    except ValueError:
        return None
    margin = config.offside.pitch_bounds_margin_m
    valid = (
        np.isfinite(pitch_points).all(axis=1)
        & (pitch_points[:, 0] >= -margin)
        & (pitch_points[:, 0] <= geometry.length + margin)
        & (pitch_points[:, 1] >= -margin)
        & (pitch_points[:, 1] <= geometry.width + margin)
    )
    if int(np.count_nonzero(valid)) < config.pose.min_legal_keypoints:
        return None
    return ProjectedPlayer(
        tracker_id=pose.tracker_id,
        role=role,
        image_points=image_points[valid],
        pitch_points=pitch_points[valid],
    )


def most_advanced_player(
    players: list[ProjectedPlayer],
    direction: AttackDirection,
) -> ProjectedPlayer | None:
    """Return the player whose legal body part is nearest the attacked goal."""

    if not players:
        return None
    return max(players, key=lambda player: direction.sign * player.reference_x(direction))


def evaluate_offside_position(
    attacker: ProjectedPlayer | None,
    opponents: list[ProjectedPlayer],
    ball_x_m: float | None,
    direction: AttackDirection,
    *,
    pitch: SoccerPitch | None = None,
    uncertainty_m: float = 0.50,
) -> OffsideDecision:
    """Apply the positional part of IFAB Law 11 at the selected kick frame.

    A positive margin means the attacker is beyond both the ball and the
    second-last opponent toward the attacked goal. Active involvement is not
    inferred, so OFFSIDE means an offside position rather than a final offence.
    """

    if attacker is None:
        return OffsideDecision(
            DecisionStatus.UNAVAILABLE,
            "No selected attacker has enough confident legal pose keypoints.",
        )
    attacker_x = attacker.reference_x(direction)
    ordered = sorted(
        opponents,
        key=lambda player: direction.sign * player.reference_x(direction),
        reverse=True,
    )
    if len(ordered) < 2:
        return OffsideDecision(
            DecisionStatus.UNAVAILABLE,
            "Fewer than two opponents have usable legal pose keypoints.",
            attacker_id=attacker.tracker_id,
        )
    second_last = ordered[1]
    defender_x = second_last.reference_x(direction)
    if ball_x_m is None or not np.isfinite(ball_x_m):
        return OffsideDecision(
            DecisionStatus.UNAVAILABLE,
            "The ball is missing or cannot be projected at the kick frame.",
            attacker_id=attacker.tracker_id,
            second_last_opponent_id=second_last.tracker_id,
            defender_line_x_m=defender_x,
        )

    line_x = (
        max(defender_x, ball_x_m)
        if direction is AttackDirection.RIGHT
        else min(defender_x, ball_x_m)
    )
    margin = direction.sign * (attacker_x - line_x)
    geometry = pitch or SoccerPitch()
    in_opponents_half = (
        attacker_x > geometry.length / 2
        if direction is AttackDirection.RIGHT
        else attacker_x < geometry.length / 2
    )
    common = {
        "attacker_id": attacker.tracker_id,
        "second_last_opponent_id": second_last.tracker_id,
        "defender_line_x_m": defender_x,
        "ball_x_m": float(ball_x_m),
        "offside_line_x_m": float(line_x),
        "margin_m": float(margin),
    }
    if not in_opponents_half:
        return OffsideDecision(
            DecisionStatus.ONSIDE,
            "The attacker is not in the opponents' half.",
            **common,
        )
    if abs(margin) <= uncertainty_m:
        return OffsideDecision(
            DecisionStatus.UNCERTAIN,
            "The margin is inside the configured single-camera uncertainty band.",
            **common,
        )
    if margin > 0:
        return OffsideDecision(
            DecisionStatus.OFFSIDE,
            "The selected legal body part is beyond both the ball and second-last opponent.",
            **common,
        )
    return OffsideDecision(
        DecisionStatus.ONSIDE,
        "The attacker is level with or behind the effective offside line.",
        **common,
    )


def _canonical_labels(
    detector: FootballDetector,
    detections: sv.Detections,
) -> list[str]:
    class_ids = (
        detections.class_id
        if detections.class_id is not None
        else np.full(len(detections), -1, dtype=int)
    )
    return [detector.canonical_label(int(class_id)) for class_id in class_ids]


def load_kick_frame_context(
    input_path: Path,
    frame_index: int,
    config: AppConfig,
    detector: FootballDetector | None = None,
) -> KickFrameContext:
    """Warm up tracking/team state and return the selected source frame."""

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if frame_index < 0:
        raise ValueError("--frame must be non-negative.")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open input video: {input_path}")
    try:
        info = read_video_info(capture, config.video.fallback_fps)
        if info.total_frames and frame_index >= info.total_frames:
            raise ValueError(
                f"Frame {frame_index} is outside a {info.total_frames}-frame video."
            )
        start_frame = max(0, frame_index - config.offside.warmup_frames)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        active_detector = detector or FootballDetector(config.detector)
        tracker = make_byte_tracker(config, info.fps)
        teams = TeamClassifier(config.teams)
        final_frame: Frame | None = None
        final_detections = sv.Detections.empty()
        final_predictions: list[TeamPrediction] = []

        for source_frame in range(start_frame, frame_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {source_frame}.")
            detections = tracker.update_with_detections(active_detector.detect(frame))
            tracker_ids = (
                detections.tracker_id
                if detections.tracker_id is not None
                else np.full(len(detections), -1, dtype=int)
            )
            predictions = teams.update(
                frame,
                detections.xyxy,
                tracker_ids,
                _canonical_labels(active_detector, detections),
                source_frame,
            )
            if source_frame == frame_index:
                final_frame = frame.copy()
                final_detections = detections
                final_predictions = predictions
            processed = source_frame - start_frame + 1
            if processed % 50 == 0:
                LOGGER.info(
                    "Prepared %d/%d context frames",
                    processed,
                    frame_index - start_frame + 1,
                )

        if final_frame is None:
            raise RuntimeError("The selected frame was not decoded.")
        return KickFrameContext(
            frame=final_frame,
            detections=final_detections,
            team_predictions=final_predictions,
            team_model_ready=teams.is_ready,
        )
    finally:
        capture.release()


def _project_ball_x(
    detector: FootballDetector,
    detections: sv.Detections,
    homography: FloatArray,
) -> float | None:
    if detections.class_id is None:
        return None
    ball_indices = [
        index
        for index, class_id in enumerate(detections.class_id)
        if detector.canonical_label(int(class_id)) == "ball"
    ]
    if not ball_indices:
        return None
    confidences = (
        detections.confidence
        if detections.confidence is not None
        else np.zeros(len(detections), dtype=float)
    )
    selected = max(ball_indices, key=lambda index: float(confidences[index]))
    if len(ball_indices) > 1:
        LOGGER.warning(
            "Detected %d ball candidates; using the highest-confidence box.",
            len(ball_indices),
        )
    box = detections.xyxy[selected]
    center = [[float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)]]
    try:
        pitch_point = project_points(center, homography)[0]
    except ValueError:
        return None
    return float(pitch_point[0]) if np.isfinite(pitch_point).all() else None


def _box_ground_anchor_on_pitch(
    box_xyxy: npt.ArrayLike,
    homography: FloatArray,
    pitch: SoccerPitch,
    margin_m: float,
) -> bool:
    """Reject people whose bounding-box contact point is outside the field."""

    box = np.asarray(box_xyxy, dtype=np.float64)
    anchor = [[float((box[0] + box[2]) / 2), float(box[3])]]
    try:
        point = project_points(anchor, homography)[0]
    except ValueError:
        return False
    return bool(
        np.isfinite(point).all()
        and -margin_m <= point[0] <= pitch.length + margin_m
        and -margin_m <= point[1] <= pitch.width + margin_m
    )


def _role_box_annotation(
    frame: Frame,
    detections: sv.Detections,
    predictions: list[TeamPrediction],
) -> Frame:
    output = frame.copy()
    tracker_ids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else np.full(len(detections), -1, dtype=int)
    )
    for box, tracker_id, prediction in zip(
        detections.xyxy,
        tracker_ids,
        predictions,
        strict=True,
    ):
        color = ROLE_COLORS[prediction.role]
        start = tuple(np.rint(box[:2]).astype(int))
        end = tuple(np.rint(box[2:]).astype(int))
        cv2.rectangle(output, start, end, color, 2)
        label = f"{prediction.role.value} #{int(tracker_id)}"
        cv2.putText(
            output,
            label,
            (start[0], max(start[1] - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _radar_observations(
    detections: sv.Detections,
    predictions: list[TeamPrediction],
    homography: FloatArray,
) -> tuple[FloatArray, list[TeamRole], list[int]]:
    tracker_ids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else np.full(len(detections), -1, dtype=int)
    )
    anchors = [
        (float((box[0] + box[2]) / 2), float(box[3]))
        for box in detections.xyxy
    ]
    points = (
        project_points(anchors, homography)
        if anchors
        else np.empty((0, 2), dtype=np.float64)
    )
    return points, [item.role for item in predictions], [int(item) for item in tracker_ids]


def _decision_banner(decision: OffsideDecision) -> tuple[str, str, tuple[int, int, int]]:
    colors = {
        DecisionStatus.OFFSIDE: (45, 45, 230),
        DecisionStatus.ONSIDE: (65, 200, 95),
        DecisionStatus.UNCERTAIN: (0, 185, 255),
        DecisionStatus.UNAVAILABLE: (145, 145, 145),
    }
    headline = {
        DecisionStatus.OFFSIDE: "OFFSIDE POSITION",
        DecisionStatus.ONSIDE: "ONSIDE",
        DecisionStatus.UNCERTAIN: "TOO CLOSE TO CALL",
        DecisionStatus.UNAVAILABLE: "DECISION UNAVAILABLE",
    }[decision.status]
    if decision.margin_m is None:
        detail = decision.reason
    else:
        detail = (
            f"Attacker #{decision.attacker_id} | margin {decision.margin_m:+.2f} m | "
            f"second-last opponent #{decision.second_last_opponent_id}"
        )
    return headline, detail, colors[decision.status]


class OffsideAnalyzer:
    """Reusable Stage 4 service with cached inference models.

    Tracking and jersey clustering are intentionally recreated for each selected
    frame because they depend on the preceding video frames. Model objects are
    retained, and a lock makes the service safe when Streamlit shares it across
    sessions.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = FootballDetector(config.detector)
        self.pitch_mapper = PitchMapper(config.pitch)
        self.pose_estimator = PoseEstimator(config.pose)
        self._lock = threading.Lock()

    def analyze(
        self,
        input_path: Path,
        frame_index: int,
        attacking_role: TeamRole,
        direction: AttackDirection,
        *,
        attacker_id: int | None = None,
        excluded_track_ids: set[int] | None = None,
    ) -> AnalysisResult:
        """Analyze one kick frame and return UI-ready views without file I/O."""

        if attacking_role not in {TeamRole.TEAM_A, TeamRole.TEAM_B}:
            raise ValueError("attacking_role must be Team A or Team B.")
        excluded = excluded_track_ids or set()
        with self._lock:
            context = load_kick_frame_context(
                input_path,
                frame_index,
                self.config,
                detector=self.detector,
            )
            self.pitch_mapper.reset()
            mapping = self.pitch_mapper.map_frame(context.frame)
            return self._analyze_context(
                input_path,
                frame_index,
                context,
                mapping,
                attacking_role,
                direction,
                attacker_id,
                excluded,
            )

    def _analyze_context(
        self,
        input_path: Path,
        frame_index: int,
        context: KickFrameContext,
        mapping: PitchMapping,
        attacking_role: TeamRole,
        direction: AttackDirection,
        attacker_id: int | None,
        excluded: set[int],
    ) -> AnalysisResult:
        """Apply pose, decision, and rendering to a prepared kick frame."""

        homography = mapping.homography
        poses: list[PlayerPose] = []
        projected_players: list[ProjectedPlayer] = []
        decision = OffsideDecision(
            DecisionStatus.UNAVAILABLE,
            "Pitch mapping is unavailable at the selected frame.",
        )
        selected_attacker: ProjectedPlayer | None = None
        if homography is not None:
            tracker_ids = (
                context.detections.tracker_id
                if context.detections.tracker_id is not None
                else np.full(len(context.detections), -1, dtype=int)
            )
            eligible_indices = [
                index
                for index, (track_id, prediction) in enumerate(
                    zip(tracker_ids, context.team_predictions, strict=True)
                )
                if int(track_id) >= 0
                and int(track_id) not in excluded
                and prediction.role
                in {TeamRole.TEAM_A, TeamRole.TEAM_B, TeamRole.GOALKEEPER}
                and _box_ground_anchor_on_pitch(
                    context.detections.xyxy[index],
                    homography,
                    self.pitch_mapper.pitch,
                    self.config.offside.pitch_bounds_margin_m,
                )
            ]
            poses = self.pose_estimator.estimate(
                context.frame,
                context.detections.xyxy[eligible_indices],
                tracker_ids[eligible_indices],
            )
            roles_by_id = {
                int(tracker_ids[index]): context.team_predictions[index].role
                for index in eligible_indices
            }
            for pose in poses:
                projected = project_player_pose(
                    pose,
                    roles_by_id[pose.tracker_id],
                    homography,
                    self.config,
                    self.pitch_mapper.pitch,
                )
                if projected is not None:
                    projected_players.append(projected)

            attackers = [
                player
                for player in projected_players
                if player.role is attacking_role
            ]
            if attacker_id is not None:
                selected_attacker = next(
                    (
                        player
                        for player in attackers
                        if player.tracker_id == attacker_id
                    ),
                    None,
                )
                if selected_attacker is None:
                    decision = OffsideDecision(
                        DecisionStatus.UNAVAILABLE,
                        f"Attacker track #{attacker_id} is not a posed "
                        f"{attacking_role.value} player.",
                        attacker_id=attacker_id,
                    )
            else:
                selected_attacker = most_advanced_player(attackers, direction)

            if selected_attacker is not None:
                defending_role = (
                    TeamRole.TEAM_B
                    if attacking_role is TeamRole.TEAM_A
                    else TeamRole.TEAM_A
                )
                opponents = [
                    player
                    for player in projected_players
                    if player.role in {defending_role, TeamRole.GOALKEEPER}
                ]
                ball_x = _project_ball_x(
                    self.detector,
                    context.detections,
                    homography,
                )
                decision = evaluate_offside_position(
                    selected_attacker,
                    opponents,
                    ball_x,
                    direction,
                    pitch=self.pitch_mapper.pitch,
                    uncertainty_m=self.config.offside.decision_uncertainty_m,
                )
            elif attacker_id is None:
                reason = (
                    "The team color model was not ready at the selected frame."
                    if not context.team_model_ready
                    else "No attacking-team player has enough legal pose keypoints."
                )
                decision = OffsideDecision(DecisionStatus.UNAVAILABLE, reason)

        base_view = _role_box_annotation(
            context.frame,
            context.detections,
            context.team_predictions,
        )
        highlighted_ids = {
            track_id
            for track_id in (
                decision.attacker_id,
                decision.second_last_opponent_id,
            )
            if track_id is not None
        }
        reference_points = {
            player.tracker_id: player.reference_image_point(direction)
            for player in projected_players
            if player.tracker_id in highlighted_ids
        }
        base_view = annotate_player_poses(
            base_view,
            poses,
            self.config.pose.keypoint_confidence,
            reference_points,
        )
        if homography is not None and decision.offside_line_x_m is not None:
            base_view = draw_pitch_x_line(
                base_view,
                homography,
                decision.offside_line_x_m,
                self.pitch_mapper.pitch,
            )
        if homography is not None:
            radar_points, radar_roles, radar_ids = _radar_observations(
                context.detections,
                context.team_predictions,
                homography,
            )
        else:
            radar_points = np.empty((0, 2), dtype=np.float64)
            radar_roles = []
            radar_ids = []
        radar = render_radar(
            radar_points,
            radar_roles,
            radar_ids,
            mapping.status,
            self.config.pitch,
            self.pitch_mapper.pitch,
            line_x_m=decision.offside_line_x_m,
            highlight_tracker_id=(
                selected_attacker.tracker_id
                if selected_attacker is not None
                else None
            ),
        )
        headline, detail, color = _decision_banner(decision)
        camera_view = draw_decision_banner(base_view, headline, detail, color)
        combined_view = overlay_radar(base_view, radar, self.config.pitch)
        combined_view = draw_decision_banner(combined_view, headline, detail, color)
        payload: dict[str, object] = {
            "input": str(input_path.expanduser().resolve()),
            "frame_index": frame_index,
            "attacking_team": attacking_role.value,
            "attack_direction": direction.value,
            "team_model_ready": context.team_model_ready,
            "pitch_mapping_status": mapping.status.value,
            "projected_player_count": len(projected_players),
            "decision": decision.as_dict(),
            "limitations": [
                "This classifies offside position, not active involvement or a final offence.",
                "Body keypoints above the ground are approximated with a planar homography.",
                "A single-camera result inside the uncertainty band is not called.",
            ],
        }
        return AnalysisResult(
            frame_index=frame_index,
            decision=decision,
            projected_player_count=len(projected_players),
            mapping_status=mapping.status,
            camera_view=camera_view,
            radar_view=radar,
            combined_view=combined_view,
            payload=payload,
        )


def write_analysis_result(
    result: AnalysisResult,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> AnalysisReport:
    """Atomically save the combined PNG and machine-readable JSON result."""

    output_path = output_path.expanduser().resolve()
    json_path = output_path.with_suffix(".json")
    for path in (output_path, json_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
    if not cv2.imwrite(str(temporary_image), result.combined_view):
        raise RuntimeError(f"Could not write analysis image: {temporary_image}")
    temporary_image.replace(output_path)
    temporary_json = json_path.with_name(f"{json_path.stem}.partial{json_path.suffix}")
    temporary_json.write_text(
        json.dumps(result.payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)
    return AnalysisReport(
        output_path=output_path,
        json_path=json_path,
        frame_index=result.frame_index,
        decision=result.decision,
        projected_player_count=result.projected_player_count,
    )


def analyze_kick_frame(
    input_path: Path,
    output_path: Path,
    frame_index: int,
    attacking_role: TeamRole,
    direction: AttackDirection,
    config: AppConfig,
    *,
    attacker_id: int | None = None,
    excluded_track_ids: set[int] | None = None,
    overwrite: bool = False,
) -> AnalysisReport:
    """Run and save the complete Stage 4 workflow for one selected frame."""

    resolved_output = output_path.expanduser().resolve()
    for path in (resolved_output, resolved_output.with_suffix(".json")):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )
    result = OffsideAnalyzer(config).analyze(
        input_path,
        frame_index,
        attacking_role,
        direction,
        attacker_id=attacker_id,
        excluded_track_ids=excluded_track_ids,
    )
    return write_analysis_result(result, resolved_output, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze offside position at a user-selected kick frame.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True, help="Zero-based kick frame.")
    parser.add_argument("--attacking-team", choices=("A", "B"), required=True)
    parser.add_argument(
        "--attack-direction",
        choices=("left", "right"),
        required=True,
        help="Direction the attacking team moves in pitch coordinates.",
    )
    parser.add_argument(
        "--attacker-id",
        type=int,
        help="Track ID to judge; defaults to the most advanced attacker.",
    )
    parser.add_argument(
        "--exclude-track",
        type=int,
        action="append",
        default=[],
        help="Track ID to exclude; repeat for known officials or bad detections.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_CONFIG.offside.warmup_frames)
    parser.add_argument(
        "--uncertainty-m",
        type=float,
        default=DEFAULT_CONFIG.offside.decision_uncertainty_m,
    )
    parser.add_argument("--model", default=DEFAULT_CONFIG.detector.model_path)
    parser.add_argument("--pose-model", default=DEFAULT_CONFIG.pose.model_path)
    parser.add_argument("--pitch-model", default=DEFAULT_CONFIG.pitch.model_path)
    parser.add_argument("--device", default=DEFAULT_CONFIG.detector.device)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG.detector.image_size)
    parser.add_argument("--pose-imgsz", type=int, default=DEFAULT_CONFIG.pose.image_size)
    parser.add_argument("--pitch-imgsz", type=int, default=DEFAULT_CONFIG.pitch.image_size)
    parser.add_argument(
        "--person-conf",
        type=float,
        default=DEFAULT_CONFIG.detector.person_confidence,
    )
    parser.add_argument("--ball-conf", type=float, default=DEFAULT_CONFIG.detector.ball_confidence)
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=DEFAULT_CONFIG.pose.detection_confidence,
    )
    parser.add_argument(
        "--pose-keypoint-conf",
        type=float,
        default=DEFAULT_CONFIG.pose.keypoint_confidence,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    detector = replace(
        DEFAULT_CONFIG.detector,
        model_path=args.model,
        device=args.device,
        image_size=args.imgsz,
        person_confidence=args.person_conf,
        ball_confidence=args.ball_conf,
    )
    pose = replace(
        DEFAULT_CONFIG.pose,
        model_path=args.pose_model,
        device=args.device,
        image_size=args.pose_imgsz,
        detection_confidence=args.pose_conf,
        keypoint_confidence=args.pose_keypoint_conf,
    )
    pitch = replace(
        DEFAULT_CONFIG.pitch,
        model_path=args.pitch_model,
        device=args.device,
        image_size=args.pitch_imgsz,
    )
    offside = replace(
        DEFAULT_CONFIG.offside,
        warmup_frames=args.warmup_frames,
        decision_uncertainty_m=args.uncertainty_m,
    )
    for name, value in (
        ("--person-conf", detector.person_confidence),
        ("--ball-conf", detector.ball_confidence),
        ("--pose-conf", pose.detection_confidence),
        ("--pose-keypoint-conf", pose.keypoint_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    for name, value in (
        ("--imgsz", detector.image_size),
        ("--pose-imgsz", pose.image_size),
        ("--pitch-imgsz", pitch.image_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if offside.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")
    if offside.decision_uncertainty_m < 0:
        raise ValueError("--uncertainty-m must be non-negative.")
    return replace(
        DEFAULT_CONFIG,
        detector=detector,
        pose=pose,
        pitch=pitch,
        offside=offside,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    attacking_role = TeamRole.TEAM_A if args.attacking_team == "A" else TeamRole.TEAM_B
    direction = AttackDirection(args.attack_direction)
    output = args.output or (
        OUTPUTS_DIR / f"{args.input.stem}_offside_frame_{args.frame}.png"
    )
    try:
        report = analyze_kick_frame(
            args.input,
            output,
            args.frame,
            attacking_role,
            direction,
            config_from_args(args),
            attacker_id=args.attacker_id,
            excluded_track_ids=set(args.exclude_track),
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info(
        "Stage 4 result: %s%s",
        report.decision.status.value,
        (
            f" ({report.decision.margin_m:+.2f} m)"
            if report.decision.margin_m is not None
            else ""
        ),
    )
    LOGGER.info("Saved image to %s", report.output_path)
    LOGGER.info("Saved machine-readable result to %s", report.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
