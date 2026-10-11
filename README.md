# Mini Football VAR

A staged personal computer-vision project for semi-automated offside review
from a single broadcast camera.

## Current status

**Stages 1-5 are implemented:** Ultralytics YOLO detection, Supervision
ByteTrack tracking, jersey-color team clustering, robust pitch homography,
temporal smoothing, a top-down radar, human pose, and single-frame offside-
position analysis, plus a Streamlit review interface for selecting the kick
frame and inspecting the result.

The default `yolo11n.pt` model is pretrained on COCO. It detects people as
`person` and the football as `sports ball`; it cannot reliably tell a player,
referee, or goalkeeper apart. The pipeline also accepts custom weights whose
class names include `player`, `goalkeeper`, `referee`, or `ball`.

## Project layout

```text
var/
  data/
    clips/       # local inputs (gitignored)
    uploads/     # Streamlit uploads by content hash (gitignored)
    weights/     # custom/downloaded weights (gitignored)
    outputs/     # generated videos (gitignored)
  src/
    config.py    # all defaults and thresholds
    detect.py    # YOLO detection and class filtering
    track.py     # ByteTrack, annotation, video I/O, CLI
    teams.py     # torso colors and stable Team A/B clustering
    pitch.py     # pitch keypoints, homography validation, temporal smoothing
    pose.py      # cropped-player YOLO pose and legal body parts
    offside.py   # kick-frame orchestration and Law 11 positional logic
    ui.py        # upload, metadata, frame, form, and download helpers
    viz.py       # annotations, radar, pose, and offside-line overlays
  tests/
    test_geometry.py
    test_teams.py
    test_pitch.py
    test_offside.py
    test_ui.py
  app.py         # Streamlit review interface
  requirements.txt
  pyproject.toml
```

## Setup

Python 3.11 is required. From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ultralytics downloads `yolo11n.pt` into `data/weights/` on the first run.
Network access is only needed for that download. For offline use, put compatible
weights in `data/weights/` and pass their path with `--model`.

Stage 3 uses the 32-landmark pitch model published by Roboflow Sports. Download
it once (the file is about 140 MB and remains gitignored):

```bash
gdown 1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf \
  -O data/weights/football-pitch-detection.pt
```

Stage 4 uses the official Ultralytics YOLO11 nano pose checkpoint:

```bash
curl -L \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt \
  -o data/weights/yolo11n-pose.pt
```

Ultralytics is distributed under AGPL-3.0 (with a separate enterprise license
available). That is normally fine for a personal project, but review its terms
before distributing or commercializing an application built around it.

## Run Stage 1

Copy a short football clip into `data/clips/`, then run:

```bash
python -m src.track --input data/clips/match.mp4
```

The annotated result is written to `data/outputs/match_tracked.mp4`. Existing
outputs are not overwritten unless you pass `--overwrite`.

Useful options:

```bash
# Quick first-run smoke test
python -m src.track --input data/clips/match.mp4 --max-frames 100

# Apple Silicon acceleration
python -m src.track --input data/clips/match.mp4 --device mps

# CUDA GPU 0 and a larger YOLO11 model
python -m src.track --input data/clips/match.mp4 --device 0 --model yolo11s.pt

# Custom football weights and a lower small-ball threshold
python -m src.track \
  --input data/clips/match.mp4 \
  --model data/weights/football.pt \
  --ball-conf 0.07
```

Run the geometry test:

```bash
pytest
```

Outputs use H.264/yuv420p with fast-start metadata for QuickTime, browser, and
Streamlit compatibility. They contain video only; source audio is not copied.

## Run Stage 2

Start with a short preview. Team labels appear after the classifier has observed
at least six stable tracks from pitch-dominant frames:

```bash
python -m src.track \
  --input data/clips/match.mp4 \
  --output data/outputs/match_teams_preview.mp4 \
  --device mps \
  --imgsz 960 \
  --start-frame 6950 \
  --max-frames 300 \
  --teams
```

Frame 6950 is a wide in-play view in the included `match.mp4`; choose a different
start for other clips. For a full clip, omit both `--start-frame` and
`--max-frames`. If `--output` is also omitted, the Stage 2 default is
`data/outputs/match_teams.mp4`.

Team A and Team B are arbitrary but stable color clusters, not home/away labels.
The classifier samples the central torso, removes grass-like pixels, represents
each track by its median CIELAB color, and fits K-means with one prototype per
track. It aligns centers between refits to prevent Team A/B from swapping.
Named `referee` and `goalkeeper` detections are excluded. With COCO weights,
those roles are all named `person`; unusual colors are marked `Unknown` only
when they are sufficiently far from both learned team centers.

## Run Stage 3

Start with the same short, wide-view segment used to validate Stage 2:

```bash
python -m src.track \
  --input data/clips/match.mp4 \
  --output data/outputs/match_radar_preview.mp4 \
  --device mps \
  --imgsz 960 \
  --start-frame 6950 \
  --max-frames 300 \
  --radar
```

Use `--device 0` on CUDA or `--device cpu` without an accelerator. `--radar`
also enables the Stage 2 team classifier. Add `--show-pitch-keypoints` to draw
the numbered landmarks used by calibration; this is useful for diagnosis but
visually busy. If `--output` is omitted, the result is
`data/outputs/match_radar.mp4`.

The pitch model predicts 32 indexed field markings. Confident landmarks are
matched to a 120 x 70 m template, `cv2.findHomography` rejects outliers with
RANSAC, and accepted transforms are smoothed across compatible frames. A bad
or missing fit shows `mapping unavailable`; a short miss shows
`calibration held` for at most five frames. Detection and tracking continue in
either case.

The radar uses each bounding box's bottom-centre as its ground contact. This is
reasonable for standing players, but an airborne ball cannot be localized on
the ground plane from one camera. The debug thresholds are centralized in
`PitchConfig` and can be overridden at the command line where appropriate:

```bash
python -m src.track --input data/clips/match.mp4 --radar \
  --pitch-model data/weights/football-pitch-detection.pt \
  --pitch-imgsz 640 --pitch-conf 0.25 --keypoint-conf 0.50
```

## Run Stage 4

First use a Stage 3 preview to identify the exact kick frame, the attacking Team
A/B label, the attacking direction in pitch coordinates, and the target track
ID. Then analyze that frame:

```bash
python -m src.offside \
  --input data/clips/match.mp4 \
  --frame 6950 \
  --attacking-team A \
  --attack-direction left \
  --attacker-id 1 \
  --device mps \
  --imgsz 960 \
  --output data/outputs/match_offside_frame_6950.png
```

The command processes the preceding 150 frames by default so ByteTrack and the
jersey classifier are ready at the kick frame. Reduce this only for a quick
test with `--warmup-frames 40`. If `--attacker-id` is omitted, the system judges
the most advanced posed attacker. Repeat `--exclude-track ID` to remove a known
official or false person detection.

YOLO11 pose predicts the 17 COCO body keypoints. The analysis retains the nose,
eyes, ears, shoulders, hips, knees, and ankles; elbows and wrists are excluded.
Shoulder centres approximate the legal torso boundary because COCO pose does
not provide the bottom-of-armpit points used by Law 11.

For the chosen attack direction, each player's most advanced legal keypoint is
projected into pitch coordinates. Opponents are ordered toward the goal to find
the second-last opponent. The effective line is whichever is nearer the goal:
the ball or second-last opponent. The attacker must also be in the opponents'
half. Margins within the default 0.50 m heuristic uncertainty band are reported
as `uncertain`, not forced to offside or onside.

Stage 4 writes an annotated PNG and a JSON file with the selected IDs, defender
line, ball position, effective line, signed margin, result, and limitations.
`offside` means **offside position only**. A final offence also requires active
involvement, which this single-frame system does not infer.

## Run Stage 5

Start the review interface from the project root:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. The workflow is:

1. Upload an MP4, MOV, M4V, MKV, or AVI clip.
2. Use the frame slider and raw-frame preview to find the instant the ball is
   played, then click **Mark kick frame**.
3. Select Team A/B, the attacking direction, and optionally an attacker track
   or excluded track IDs.
4. Click **Run offside analysis**.
5. Review the annotated camera frame and top-down radar side by side, then
   download the combined PNG and JSON evidence.

Uploaded clips are stored locally under `data/uploads/` using a content hash;
that directory is gitignored. The upload limit is configured to 1 GB in
`.streamlit/config.toml`. The app prefers Apple MPS when available and exposes
CPU/automatic device choices under **Runtime settings**. The default 150-frame
tracking warm-up is more stable but slower; reducing it is useful for a quick
smoke test.

Detection, pitch, and pose model objects are cached across Streamlit reruns.
Tracking, team clustering, and homography state are recreated for every marked
frame so one review cannot contaminate another. Failures such as a missing ball,
bad calibration, or inadequate pose evidence are shown as a reasoned
`decision unavailable` state instead of crashing the app.

## Detection quality and fine-tuning

Start with pretrained weights to validate the pipeline. Fine-tuning is likely
needed for the ball because it occupies very few pixels, is motion-blurred, and
is frequently occluded. A practical next dataset is a football-specific object
detection dataset from **Roboflow Universe**, provided its license and labels
fit the project. **SoccerNet** is the stronger research-oriented source for
broadcast football data; use its tracking/game-state resources after accepting
its access and license terms. Keep train/validation splits separated by match,
not random frames, to avoid leakage.

Do not fine-tune yet unless the default model fails on representative clips:
first collect false negatives and measure ball recall. If training becomes
necessary, use 1280px or tiled inputs, include hard negatives (white markings,
boots, advertising), and retain people/referee labels in the custom model.

Pitch fine-tuning is separate from ball fine-tuning. It is needed if the radar
frequently reports `mapping unavailable`, numbered landmarks visibly attach to
the wrong field intersections, or calibration fails on the broadcast style you
care about. Start from Roboflow's public
[`football-field-detection-f07vi/14`](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/14)
32-keypoint dataset. For research use, SoccerNet also publishes camera
calibration annotations subject to its access terms. Split training and
validation by match/camera, never by random adjacent frames.

## Expected failure modes

- A missing ball detection is treated as normal and processing continues.
- ByteTrack can change IDs after a camera cut, a long occlusion, or players
  crossing tightly. Process one continuous broadcast shot at a time.
- Team labels remain `Unknown` until enough persistent tracks have been sampled
  on a pitch-dominant view; close-ups and title cards are intentionally ignored.
- COCO labels all people alike. Referee/goalkeeper exclusion therefore remains
  a color-outlier heuristic unless custom weights provide those named classes.
- Very large inference sizes improve small-ball recall but cost substantial
  speed and memory.
- Pitch mapping needs at least five confident, non-degenerate landmarks. It is
  intentionally withheld on close-ups, replays, severe occlusion, and graphics.
- Camera cuts and large pans reset smoothing. Up to five missing fits reuse the
  last transform; longer gaps make the radar unavailable instead of inventing
  coordinates.
- Missing ball, fewer than two posed opponents, a missing selected attacker, or
  unavailable pitch calibration produces `decision unavailable`.
- Distant players may not yield enough confident COCO pose keypoints even after
  per-player cropping. Lowering the confidence threshold can add wrong joints.
- Generic COCO detections can classify a referee as a player or a false object
  as the ball. Use `--exclude-track` and football-specific weights where needed.
- A corrupt clip, invalid frame size, unavailable encoder, unsupported model,
  or existing output produces a clear error instead of a partial final file.

## Accuracy limits of a single broadcast camera

This project can be a useful review aid, not a replacement for calibrated
multi-camera VAR. A single moving/zooming camera has occlusions, rolling shutter,
motion blur, uncertain pitch calibration, and no direct depth. The exact kick
instant and the most advanced legal body part can each shift the result. Even
after all stages, close decisions should be reported with an uncertainty band;
sub-centimetre claims would be misleading.

The pitch template is 120 x 70 m because that is the landmark convention used
by the pretrained model. Real professional pitches vary in size, and broadcast
lens distortion is not represented by one planar homography. Stage 4 reports an
experimental margin, not a survey-grade measurement. Actual pitch dimensions
and measured calibration uncertainty are required for stronger claims.

Projecting a head or torso keypoint through a ground-plane homography is also an
approximation because the point is above the grass. The fixed 0.50 m uncertainty
band prevents forced close calls but does not fully quantify camera, pose,
timing, or lens error.

## Roadmap

1. **Detection + tracking** — complete.
2. **Team split** — complete; torso colors, stable clustering, role exclusions.
3. **Pitch mapping** — complete; keypoints, robust/smoothed homography, radar.
4. **Pose + offside** — complete; selected kick frame, legal parts, second-last
   opponent, ball-aware line, signed margin, and uncertainty state.
5. **Streamlit UI** — complete; upload, scrubber, marked kick frame, cached
   analysis, dual visualization, failure states, and PNG/JSON downloads.
