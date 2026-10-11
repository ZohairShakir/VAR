"""Video overlays and the Stage 3 top-down pitch radar."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from src.config import PitchConfig
from src.pitch import MappingStatus, PitchKeypoints, SoccerPitch, project_points
from src.pose import LEGAL_SKELETON_EDGES, PlayerPose
from src.teams import TeamRole


Frame = npt.NDArray[np.uint8]

ROLE_COLORS: dict[TeamRole, tuple[int, int, int]] = {
    TeamRole.TEAM_A: (235, 99, 37),
    TeamRole.TEAM_B: (38, 38, 220),
    TeamRole.REFEREE: (11, 158, 245),
    TeamRole.GOALKEEPER: (247, 85, 168),
    TeamRole.UNKNOWN: (184, 163, 148),
    TeamRole.BALL: (21, 204, 250),
}


def _pitch_to_pixels(
    points_xy: npt.ArrayLike,
    pitch: SoccerPitch,
    canvas_width: int,
    padding: int,
) -> tuple[npt.NDArray[np.int32], int]:
    usable_width = canvas_width - 2 * padding
    scale = usable_width / pitch.length
    canvas_height = int(round(pitch.width * scale)) + 2 * padding
    points = np.asarray(points_xy, dtype=np.float64)
    pixels = np.column_stack(
        (
            padding + points[:, 0] * scale,
            padding + points[:, 1] * scale,
        )
    )
    return np.rint(pixels).astype(np.int32), canvas_height


def render_radar(
    pitch_points: npt.ArrayLike,
    roles: list[TeamRole],
    tracker_ids: list[int],
    status: MappingStatus,
    config: PitchConfig,
    pitch: SoccerPitch | None = None,
    *,
    line_x_m: float | None = None,
    highlight_tracker_id: int | None = None,
) -> Frame:
    """Render a scaled pitch with projected player/ball locations."""

    geometry = pitch or SoccerPitch()
    padding = 16
    vertices_px, pitch_height = _pitch_to_pixels(
        geometry.vertices,
        geometry,
        config.radar_width,
        padding,
    )
    title_height = 28
    canvas = np.full(
        (pitch_height + title_height, config.radar_width, 3),
        (26, 55, 34),
        dtype=np.uint8,
    )
    field = canvas[title_height:]

    field_left = padding
    field_right = config.radar_width - padding
    field_top = padding
    field_bottom = pitch_height - padding
    stripe_width = max((field_right - field_left) // 12, 1)
    for stripe in range(12):
        color = (52, 116, 67) if stripe % 2 == 0 else (46, 105, 61)
        start = field_left + stripe * stripe_width
        end = field_right if stripe == 11 else start + stripe_width
        cv2.rectangle(field, (start, field_top), (end, field_bottom), color, -1)

    for start, end in geometry.edges:
        cv2.line(
            field,
            tuple(vertices_px[start]),
            tuple(vertices_px[end]),
            (235, 242, 236),
            1,
            cv2.LINE_AA,
        )
    scale = (config.radar_width - 2 * padding) / geometry.length
    centre = (
        int(round(padding + geometry.length * scale / 2)),
        int(round(padding + geometry.width * scale / 2)),
    )
    cv2.circle(
        field,
        centre,
        max(int(round(geometry.centre_circle_radius * scale)), 1),
        (235, 242, 236),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(field, centre, 2, (235, 242, 236), -1, cv2.LINE_AA)
    for penalty_index in (8, 21):
        cv2.circle(
            field,
            tuple(vertices_px[penalty_index]),
            2,
            (235, 242, 236),
            -1,
            cv2.LINE_AA,
        )

    if line_x_m is not None and np.isfinite(line_x_m):
        line_x = int(round(padding + line_x_m * scale))
        if field_left <= line_x <= field_right:
            cv2.line(
                field,
                (line_x, field_top),
                (line_x, field_bottom),
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

    points = np.asarray(pitch_points, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pitch_points must have shape (N, 2).")
    if len(points) != len(roles) or len(points) != len(tracker_ids):
        raise ValueError("Pitch points, roles, and tracker IDs must align.")
    points_px, _ = _pitch_to_pixels(
        points,
        geometry,
        config.radar_width,
        padding,
    )
    for point, pixel, role, tracker_id in zip(
        points,
        points_px,
        roles,
        tracker_ids,
        strict=True,
    ):
        if not (
            np.isfinite(point).all()
            and 0.0 <= point[0] <= geometry.length
            and 0.0 <= point[1] <= geometry.width
        ):
            continue
        color = ROLE_COLORS[role]
        radius = 5 if role is TeamRole.BALL else 7
        cv2.circle(field, tuple(pixel), radius + 2, (15, 23, 18), -1, cv2.LINE_AA)
        cv2.circle(field, tuple(pixel), radius, color, -1, cv2.LINE_AA)
        if tracker_id == highlight_tracker_id:
            cv2.circle(
                field,
                tuple(pixel),
                radius + 5,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
        if tracker_id >= 0 and role is not TeamRole.BALL:
            cv2.putText(
                field,
                str(tracker_id),
                (int(pixel[0]) + 7, int(pixel[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    status_color = {
        MappingStatus.FRESH: (112, 235, 158),
        MappingStatus.SMOOTHED: (112, 235, 158),
        MappingStatus.RESET: (86, 201, 250),
        MappingStatus.HELD: (86, 201, 250),
        MappingStatus.UNAVAILABLE: (107, 114, 128),
    }[status]
    status_text = {
        MappingStatus.FRESH: "calibrated",
        MappingStatus.SMOOTHED: "calibrated / smoothed",
        MappingStatus.RESET: "calibrated / reset",
        MappingStatus.HELD: "calibration held",
        MappingStatus.UNAVAILABLE: "mapping unavailable",
    }[status]
    cv2.putText(
        canvas,
        "PITCH RADAR",
        (12, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    text_size = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        1,
    )[0]
    cv2.putText(
        canvas,
        status_text,
        (config.radar_width - text_size[0] - 10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        status_color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def overlay_radar(frame: Frame, radar: Frame, config: PitchConfig) -> Frame:
    """Blend the radar into the upper-right corner of a broadcast frame."""

    output = frame.copy()
    margin = max(config.radar_margin, 0)
    max_width = frame.shape[1] - 2 * margin
    max_height = frame.shape[0] - 2 * margin
    if max_width <= 0 or max_height <= 0:
        return output
    scale = min(1.0, max_width / radar.shape[1], max_height / radar.shape[0])
    if scale < 1.0:
        radar = cv2.resize(
            radar,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    x1 = frame.shape[1] - margin - radar.shape[1]
    y1 = margin
    x2 = x1 + radar.shape[1]
    y2 = y1 + radar.shape[0]
    region = output[y1:y2, x1:x2]
    cv2.addWeighted(
        radar,
        config.radar_opacity,
        region,
        1.0 - config.radar_opacity,
        0.0,
        dst=region,
    )
    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (222, 229, 224), 1)
    return output


def annotate_pitch_keypoints(
    frame: Frame,
    keypoints: PitchKeypoints | None,
    confidence_threshold: float,
) -> Frame:
    """Draw indexed keypoints for calibration debugging."""

    output = frame.copy()
    if keypoints is None:
        return output
    for index, (point, confidence) in enumerate(
        zip(keypoints.xy, keypoints.confidence, strict=True),
        start=1,
    ):
        if confidence < confidence_threshold or not np.isfinite(point).all():
            continue
        center = tuple(np.rint(point).astype(int))
        cv2.circle(output, center, 4, (255, 80, 220), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(index),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def annotate_player_poses(
    frame: Frame,
    poses: list[PlayerPose],
    confidence_threshold: float,
    reference_points: dict[int, npt.NDArray[np.float64]] | None = None,
) -> Frame:
    """Draw only offside-relevant pose joints and skeleton connections."""

    output = frame.copy()
    references = reference_points or {}
    for pose in poses:
        valid = pose.legal_mask(confidence_threshold)
        for start, end in LEGAL_SKELETON_EDGES:
            if not (valid[start] and valid[end]):
                continue
            first = tuple(np.rint(pose.keypoints_xy[start]).astype(int))
            second = tuple(np.rint(pose.keypoints_xy[end]).astype(int))
            cv2.line(output, first, second, (80, 230, 120), 2, cv2.LINE_AA)
        for index in np.flatnonzero(valid):
            point = tuple(np.rint(pose.keypoints_xy[index]).astype(int))
            cv2.circle(output, point, 3, (80, 230, 120), -1, cv2.LINE_AA)
        reference = references.get(pose.tracker_id)
        if reference is not None and np.isfinite(reference).all():
            point = tuple(np.rint(reference).astype(int))
            cv2.circle(output, point, 10, (0, 215, 255), 2, cv2.LINE_AA)
    return output


def draw_pitch_x_line(
    frame: Frame,
    image_to_pitch: npt.ArrayLike,
    line_x_m: float,
    pitch: SoccerPitch | None = None,
) -> Frame:
    """Draw a constant pitch-x line back in broadcast-image perspective."""

    output = frame.copy()
    geometry = pitch or SoccerPitch()
    try:
        pitch_to_image = np.linalg.inv(np.asarray(image_to_pitch, dtype=np.float64))
        endpoints = project_points(
            [[line_x_m, 0.0], [line_x_m, geometry.width]],
            pitch_to_image,
        )
    except (ValueError, np.linalg.LinAlgError):
        return output
    points = np.rint(endpoints).astype(int)
    rectangle = (0, 0, output.shape[1], output.shape[0])
    visible, start, end = cv2.clipLine(
        rectangle,
        tuple(points[0]),
        tuple(points[1]),
    )
    if visible:
        cv2.line(output, start, end, (0, 215, 255), 3, cv2.LINE_AA)
    return output


def draw_decision_banner(
    frame: Frame,
    headline: str,
    detail: str,
    color: tuple[int, int, int],
) -> Frame:
    """Draw a readable Stage 4 result banner without hiding the radar."""

    output = frame.copy()
    panel_width = min(max(output.shape[1] // 2, 440), output.shape[1] - 24)
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width, 82), (18, 24, 21), -1)
    cv2.addWeighted(overlay, 0.90, output, 0.10, 0.0, dst=output)
    cv2.rectangle(output, (12, 12), (12 + panel_width, 82), color, 2)
    cv2.putText(
        output,
        headline,
        (26, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        detail[:90],
        (26, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 244, 241),
        1,
        cv2.LINE_AA,
    )
    return output
