"""Framework-independent tests for Stage 5 upload and form helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ui import (
    encode_json,
    encode_png,
    frame_timestamp,
    parse_optional_track_id,
    parse_track_ids,
    persist_uploaded_video,
)


def test_persist_uploaded_video_uses_safe_content_name(tmp_path: Path) -> None:
    first = persist_uploaded_video("../../match.MP4", b"video-bytes", tmp_path)
    second = persist_uploaded_video("renamed.mp4", b"video-bytes", tmp_path)

    assert first == second
    assert first.parent == tmp_path
    assert first.suffix == ".mp4"
    assert "match" not in first.name
    assert first.read_bytes() == b"video-bytes"


def test_persist_uploaded_video_rejects_empty_or_unsupported_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        persist_uploaded_video("notes.txt", b"not-video", tmp_path)
    with pytest.raises(ValueError, match="empty"):
        persist_uploaded_video("match.mp4", b"", tmp_path)


def test_track_id_parsing_accepts_common_separators() -> None:
    assert parse_track_ids("1, 4; 7  9") == {1, 4, 7, 9}
    assert parse_track_ids("") == set()
    assert parse_optional_track_id(" 12 ") == 12
    assert parse_optional_track_id(" ") is None

    with pytest.raises(ValueError, match="whole numbers"):
        parse_track_ids("1, goalkeeper")
    with pytest.raises(ValueError, match="non-negative"):
        parse_optional_track_id("-2")


def test_timestamp_and_download_encoders() -> None:
    assert frame_timestamp(75, 25.0) == "00:00:03.000"
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    assert encode_png(frame).startswith(b"\x89PNG\r\n\x1a\n")
    assert encode_json({"status": "onside"}).endswith(b"\n")
