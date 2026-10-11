"""Small, framework-independent helpers for the Stage 5 review interface."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


Frame = npt.NDArray[np.uint8]
SUPPORTED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Validated metadata used by the frame scrubber."""

    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.frame_count else 0.0


def persist_uploaded_video(
    original_name: str,
    content: bytes,
    upload_directory: Path,
) -> Path:
    """Store an upload under a content-derived, path-safe filename."""

    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise ValueError(f"Unsupported video type {suffix or '(none)'}. Use: {allowed}.")
    if not content:
        raise ValueError("The uploaded video is empty.")
    digest = hashlib.sha256(content).hexdigest()[:20]
    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / f"{digest}{suffix}"
    if destination.exists() and destination.stat().st_size == len(content):
        return destination
    temporary = destination.with_suffix(f".partial{suffix}")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return destination


def read_video_metadata(path: Path, fallback_fps: float = 30.0) -> VideoMetadata:
    """Read video metadata with the same FPS fallback as the processing pipeline."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the uploaded video.")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError("The uploaded video reports an invalid frame size.")
    if not np.isfinite(fps) or fps <= 0:
        fps = fallback_fps
    if frame_count <= 0:
        raise ValueError("The uploaded video does not report a usable frame count.")
    return VideoMetadata(width, height, fps, frame_count)


def read_video_frame(path: Path, frame_index: int) -> Frame:
    """Decode one zero-based source frame for the scrubber preview."""

    if frame_index < 0:
        raise ValueError("Frame index must be non-negative.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the uploaded video.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_index}.")
    return frame


def parse_track_ids(value: str) -> set[int]:
    """Parse optional comma/space-separated non-negative tracker IDs."""

    stripped = value.strip()
    if not stripped:
        return set()
    tokens = [token for token in re.split(r"[\s,;]+", stripped) if token]
    try:
        identifiers = {int(token) for token in tokens}
    except ValueError as exc:
        raise ValueError("Track IDs must be whole numbers separated by commas.") from exc
    if any(identifier < 0 for identifier in identifiers):
        raise ValueError("Track IDs must be non-negative.")
    return identifiers


def parse_optional_track_id(value: str) -> int | None:
    """Parse one optional non-negative tracker ID."""

    stripped = value.strip()
    if not stripped:
        return None
    try:
        identifier = int(stripped)
    except ValueError as exc:
        raise ValueError("Attacker ID must be one whole number.") from exc
    if identifier < 0:
        raise ValueError("Attacker ID must be non-negative.")
    return identifier


def frame_timestamp(frame_index: int, fps: float) -> str:
    """Format a zero-based frame index as HH:MM:SS.mmm."""

    if frame_index < 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("A non-negative frame and positive FPS are required.")
    milliseconds = int(round(frame_index * 1000.0 / fps))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def encode_png(frame: Frame) -> bytes:
    """Encode one BGR OpenCV frame for a Streamlit download button."""

    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("OpenCV could not encode the analysis image.")
    return encoded.tobytes()


def encode_json(payload: dict[str, object]) -> bytes:
    """Encode a stable, human-readable JSON download."""

    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")
