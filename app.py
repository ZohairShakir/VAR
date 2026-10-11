"""Stage 5 Streamlit interface for selecting and reviewing a kick frame."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st
import torch

from src.config import DEFAULT_CONFIG, UPLOADS_DIR, AppConfig
from src.offside import (
    AnalysisResult,
    AttackDirection,
    DecisionStatus,
    OffsideAnalyzer,
)
from src.teams import TeamRole
from src.ui import (
    Frame,
    VideoMetadata,
    encode_json,
    encode_png,
    frame_timestamp,
    parse_optional_track_id,
    parse_track_ids,
    persist_uploaded_video,
    read_video_frame,
    read_video_metadata,
)


st.set_page_config(
    page_title="Mini Football VAR",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
:root {
  --var-bg: #0b1115;
  --var-surface: #10191f;
  --var-surface-2: #142029;
  --var-border: #26343d;
  --var-text: #f4f7f8;
  --var-muted: #9aabb5;
  --var-amber: #f6b73c;
}
.stApp { background: var(--var-bg); color: var(--var-text); }
[data-testid="stHeader"] { background: rgba(11, 17, 21, 0.94); }
[data-testid="stSidebar"] { background: #0d151a; border-right: 1px solid var(--var-border); }
[data-testid="stSidebar"] hr { border-color: var(--var-border); }
.block-container { max-width: 1600px; padding-top: 1.7rem; padding-bottom: 3rem; }
.var-header { border-bottom: 1px solid var(--var-border); padding: 0 0 1.25rem; margin-bottom: 1.4rem; }
.var-header h1 { color: var(--var-text); font-size: clamp(2rem, 3vw, 3.2rem); letter-spacing: -0.045em; line-height: 1; margin: 0; }
.var-header p { color: var(--var-muted); font-size: 1rem; margin: .65rem 0 0; }
.var-section-title { color: var(--var-text); font-size: 1rem; font-weight: 700; letter-spacing: .01em; margin: .2rem 0 .65rem; }
.var-frame-meta { color: var(--var-muted); font-size: .86rem; margin: -.35rem 0 .75rem; }
.var-limit { border-top: 1px solid var(--var-border); color: var(--var-muted); font-size: .82rem; line-height: 1.55; margin-top: 1.3rem; padding-top: 1rem; }
.var-empty { border: 1px dashed #31424d; border-radius: 10px; color: var(--var-muted); padding: 4rem 2rem; text-align: center; }
div[data-testid="stMetric"] { border-left: 1px solid var(--var-border); padding-left: .9rem; }
div[data-testid="stMetricLabel"] { color: var(--var-muted); }
div[data-testid="stMetricValue"] { color: var(--var-text); font-size: 1.35rem; }
.stButton > button[kind="primary"] {
  background: var(--var-amber); border-color: var(--var-amber); color: #171107; font-weight: 750;
}
.stButton > button[kind="primary"]:hover { background: #ffc95a; border-color: #ffc95a; color: #171107; }
[data-testid="stFileUploaderDropzone"] { background: var(--var-surface); border-color: #344650; }
div[data-testid="stImage"] img, video { border-radius: 9px; border: 1px solid var(--var-border); }
@media (max-width: 760px) {
  .block-container { padding: 1rem .9rem 2rem; }
  .var-header h1 { font-size: 2rem; }
  div[data-testid="stMetric"] { margin-bottom: .7rem; }
}
</style>
"""


def _configured_app(device: str, warmup_frames: int) -> AppConfig:
    """Apply UI-selectable runtime settings without mutating global defaults."""

    model_device = None if device == "auto" else device
    return replace(
        DEFAULT_CONFIG,
        detector=replace(DEFAULT_CONFIG.detector, device=model_device),
        pitch=replace(DEFAULT_CONFIG.pitch, device=model_device),
        pose=replace(DEFAULT_CONFIG.pose, device=model_device),
        offside=replace(DEFAULT_CONFIG.offside, warmup_frames=warmup_frames),
    )


@st.cache_resource(show_spinner=False)
def _get_analyzer(device: str, warmup_frames: int) -> OffsideAnalyzer:
    """Load heavy YOLO models once per runtime configuration."""

    return OffsideAnalyzer(_configured_app(device, warmup_frames))


@st.cache_data(show_spinner=False)
def _metadata_for(path_string: str) -> VideoMetadata:
    return read_video_metadata(Path(path_string), DEFAULT_CONFIG.video.fallback_fps)


@st.cache_data(show_spinner=False, max_entries=24)
def _preview_for(path_string: str, frame_index: int) -> Frame:
    return read_video_frame(Path(path_string), frame_index)


def _device_choices() -> tuple[list[str], int]:
    choices = ["auto", "cpu"]
    preferred = 0
    if torch.backends.mps.is_available():
        choices.insert(0, "mps")
        preferred = 0
    elif torch.cuda.is_available():
        choices.insert(0, "0")
        preferred = 0
    return choices, preferred


def _reset_video_state(video_key: str) -> None:
    if st.session_state.get("video_key") == video_key:
        return
    st.session_state.video_key = video_key
    st.session_state.scrub_frame = 0
    st.session_state.marked_frame = None
    st.session_state.analysis_result = None
    st.session_state.analysis_signature = None


def _analysis_signature(
    video_key: str,
    frame_index: int,
    attacking_team: str,
    direction: str,
    attacker_text: str,
    excluded_text: str,
    device: str,
    warmup_frames: int,
) -> tuple[object, ...]:
    return (
        video_key,
        frame_index,
        attacking_team,
        direction,
        attacker_text.strip(),
        excluded_text.strip(),
        device,
        warmup_frames,
    )


def _show_result(result: AnalysisResult, file_stem: str) -> None:
    """Render the Stage 4 evidence and downloads."""

    st.markdown('<div class="var-section-title">Analysis result</div>', unsafe_allow_html=True)
    status_renderers = {
        DecisionStatus.OFFSIDE: st.error,
        DecisionStatus.ONSIDE: st.success,
        DecisionStatus.UNCERTAIN: st.warning,
        DecisionStatus.UNAVAILABLE: st.info,
    }
    verdict = {
        DecisionStatus.OFFSIDE: "OFFSIDE POSITION",
        DecisionStatus.ONSIDE: "ONSIDE",
        DecisionStatus.UNCERTAIN: "TOO CLOSE TO CALL",
        DecisionStatus.UNAVAILABLE: "DECISION UNAVAILABLE",
    }[result.decision.status]
    status_renderers[result.decision.status](f"**{verdict}** — {result.decision.reason}")

    camera_column, radar_column = st.columns((1.7, 1.0), gap="medium")
    with camera_column:
        st.markdown('<div class="var-section-title">Camera decision view</div>', unsafe_allow_html=True)
        st.image(result.camera_view, channels="BGR", width="stretch")
    with radar_column:
        st.markdown('<div class="var-section-title">Top-down radar</div>', unsafe_allow_html=True)
        st.image(result.radar_view, channels="BGR", width="stretch")

    decision = result.decision
    margin = "—" if decision.margin_m is None else f"{decision.margin_m:+.2f} m"
    attacker = "—" if decision.attacker_id is None else f"#{decision.attacker_id}"
    defender = (
        "—"
        if decision.second_last_opponent_id is None
        else f"#{decision.second_last_opponent_id}"
    )
    reference = "—"
    if decision.offside_line_x_m is not None:
        reference = (
            "Ball"
            if decision.ball_x_m is not None
            and abs(decision.offside_line_x_m - decision.ball_x_m) < 1e-6
            else "Second-last opponent"
        )
    metrics = st.columns(5)
    metrics[0].metric("Verdict", result.decision.status.value.upper())
    metrics[1].metric("Signed margin", margin)
    metrics[2].metric("Attacker track", attacker)
    metrics[3].metric("Second-last defender", defender)
    metrics[4].metric("Effective reference", reference)
    st.caption(
        f"Pitch mapping: {result.mapping_status.value} · "
        f"Projected players: {result.projected_player_count}"
    )

    png_data = encode_png(result.combined_view)
    json_data = encode_json(result.payload)
    download_png, download_json, _ = st.columns((1, 1, 3))
    with download_png:
        st.download_button(
            "Download PNG",
            png_data,
            file_name=f"{file_stem}_offside.png",
            mime="image/png",
            width="stretch",
        )
    with download_json:
        st.download_button(
            "Download JSON",
            json_data,
            file_name=f"{file_stem}_offside.json",
            mime="application/json",
            width="stretch",
        )
    st.markdown(
        '<div class="var-limit"><strong>Decision boundary:</strong> this estimates '
        'offside position at the marked frame. It does not determine active involvement '
        'or replace calibrated multi-camera VAR. Close single-camera margins remain '
        'uncertain.</div>',
        unsafe_allow_html=True,
    )


def _show_source_and_preview(
    video_path: Path,
    selected_frame: int,
    metadata: VideoMetadata,
) -> None:
    """Show playback and the exact raw frame currently selected by the scrubber."""

    source_column, preview_column = st.columns((1.25, 1.0), gap="medium")
    with source_column:
        st.markdown(
            '<div class="var-section-title">Source clip</div>',
            unsafe_allow_html=True,
        )
        st.video(str(video_path))
    with preview_column:
        st.markdown(
            '<div class="var-section-title">Selected kick-frame</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="var-frame-meta">Frame {selected_frame:,} · '
            f'{frame_timestamp(selected_frame, metadata.fps)}</div>',
            unsafe_allow_html=True,
        )
        try:
            preview = _preview_for(str(video_path), selected_frame)
            st.image(preview, channels="BGR", width="stretch")
        except RuntimeError as exc:
            st.warning(str(exc))


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="var-header"><h1>Mini Football VAR</h1>'
        '<p>Single-camera decision support</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Review setup")
        uploaded = st.file_uploader(
            "Match video",
            type=["mp4", "mov", "m4v", "mkv", "avi"],
            help="Use one continuous broadcast shot when possible. Uploads are stored locally under data/uploads.",
        )

    if uploaded is None:
        st.markdown(
            '<div class="var-empty"><strong>Upload a football clip to begin.</strong><br>'
            'You will scrub to the pass, mark the kick frame, and run one-frame offside analysis.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        video_path = persist_uploaded_video(
            uploaded.name,
            uploaded.getvalue(),
            UPLOADS_DIR,
        )
        metadata = _metadata_for(str(video_path))
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not prepare this video: {exc}")
        return

    _reset_video_state(video_path.name)
    with st.sidebar:
        duration = frame_timestamp(metadata.frame_count - 1, metadata.fps)
        st.caption(
            f"{metadata.width}×{metadata.height} · {metadata.fps:.2f} FPS  \n"
            f"{metadata.frame_count:,} frames · {duration}"
        )
        st.divider()
        st.subheader("Frame control")
        selected_frame = st.slider(
            "Kick-frame candidate",
            min_value=0,
            max_value=metadata.frame_count - 1,
            key="scrub_frame",
            help="Frame numbers are zero-based.",
        )
        st.caption(
            f"Frame {selected_frame:,} · "
            f"{frame_timestamp(selected_frame, metadata.fps)}"
        )
        if st.button("Mark kick frame", type="primary", width="stretch"):
            st.session_state.marked_frame = selected_frame
            st.session_state.analysis_result = None
            st.session_state.analysis_signature = None

        marked_frame = st.session_state.get("marked_frame")
        if marked_frame is None:
            st.info("Scrub to the pass and mark the exact contact frame.")
        else:
            st.success(
                f"Marked frame {marked_frame:,} · "
                f"{frame_timestamp(marked_frame, metadata.fps)}"
            )

        st.divider()
        attacking_team = st.radio("Attacking team", ("A", "B"), horizontal=True)
        direction_label = st.radio(
            "Attack direction",
            ("Left → Right", "Right → Left"),
        )
        attacker_text = st.text_input(
            "Attacker ID (optional)",
            placeholder="Auto: most advanced attacker",
        )
        excluded_text = st.text_input(
            "Excluded track IDs (optional)",
            placeholder="e.g. 4, 12",
            help="Use this for a known referee, goalkeeper error, or false person detection.",
        )
        with st.expander("Runtime settings"):
            device_options, preferred_device = _device_choices()
            device = st.selectbox(
                "Inference device",
                device_options,
                index=preferred_device,
                help="MPS is preferred on supported Apple Silicon Macs.",
            )
            warmup_frames = st.number_input(
                "Tracking warm-up frames",
                min_value=0,
                max_value=600,
                value=DEFAULT_CONFIG.offside.warmup_frames,
                step=10,
                help="More context stabilizes IDs and jersey clustering but takes longer.",
            )
        run_analysis = st.button(
            "Run offside analysis",
            type="primary",
            width="stretch",
            disabled=marked_frame is None,
        )

    if marked_frame is None:
        _show_source_and_preview(video_path, selected_frame, metadata)
        st.info("Mark the moment the ball is played to enable analysis.")
        return

    current_signature = _analysis_signature(
        video_path.name,
        marked_frame,
        attacking_team,
        direction_label,
        attacker_text,
        excluded_text,
        device,
        int(warmup_frames),
    )
    if run_analysis:
        try:
            attacker_id = parse_optional_track_id(attacker_text)
            excluded_ids = parse_track_ids(excluded_text)
            role = TeamRole.TEAM_A if attacking_team == "A" else TeamRole.TEAM_B
            direction = (
                AttackDirection.RIGHT
                if direction_label == "Left → Right"
                else AttackDirection.LEFT
            )
            with st.spinner("Analyzing the marked frame and replaying its context…"):
                analyzer = _get_analyzer(device, int(warmup_frames))
                result = analyzer.analyze(
                    video_path,
                    marked_frame,
                    role,
                    direction,
                    attacker_id=attacker_id,
                    excluded_track_ids=excluded_ids,
                )
            st.session_state.analysis_result = result
            st.session_state.analysis_signature = current_signature
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            st.session_state.analysis_result = None
            st.session_state.analysis_signature = None
            st.error(f"Analysis could not be completed: {exc}")

    result = st.session_state.get("analysis_result")
    if result is not None:
        if st.session_state.get("analysis_signature") != current_signature:
            st.warning("The review settings changed. Run the analysis again to refresh the result.")
        st.markdown(
            f'<div class="var-frame-meta">Selected kick frame {marked_frame:,} · '
            f'{frame_timestamp(marked_frame, metadata.fps)}</div>',
            unsafe_allow_html=True,
        )
        _show_result(result, Path(uploaded.name).stem)
        with st.expander("Source clip and raw frame preview"):
            _show_source_and_preview(video_path, selected_frame, metadata)
    else:
        _show_source_and_preview(video_path, selected_frame, metadata)
        st.markdown(
            '<div class="var-empty"><strong>The kick frame is marked.</strong><br>'
            'Confirm the attacking team and direction, then run the analysis.'
            '</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
