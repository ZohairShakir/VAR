"""Rule and legal-body-part tests for Stage 4."""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_CONFIG
from src.offside import (
    AttackDirection,
    DecisionStatus,
    ProjectedPlayer,
    evaluate_offside_position,
    project_player_pose,
)
from src.pose import EXCLUDED_ARM_INDICES, PlayerPose
from src.teams import TeamRole


def _player(track_id: int, role: TeamRole, x: float) -> ProjectedPlayer:
    points = np.array([[x, 30.0], [x - 0.2, 31.0]], dtype=np.float64)
    return ProjectedPlayer(track_id, role, points.copy(), points)


def test_right_attacking_offside_uses_second_last_opponent() -> None:
    attacker = _player(7, TeamRole.TEAM_A, 98.0)
    opponents = [
        _player(20, TeamRole.TEAM_B, 100.0),
        _player(21, TeamRole.TEAM_B, 95.0),
        _player(22, TeamRole.TEAM_B, 80.0),
    ]

    decision = evaluate_offside_position(
        attacker,
        opponents,
        ball_x_m=92.0,
        direction=AttackDirection.RIGHT,
        uncertainty_m=0.25,
    )

    assert decision.status is DecisionStatus.OFFSIDE
    assert decision.second_last_opponent_id == 21
    assert decision.offside_line_x_m == 95.0
    assert decision.margin_m == 3.0


def test_ball_can_be_nearer_goal_than_second_last_opponent() -> None:
    attacker = _player(7, TeamRole.TEAM_A, 98.0)
    opponents = [
        _player(20, TeamRole.TEAM_B, 100.0),
        _player(21, TeamRole.TEAM_B, 95.0),
    ]

    decision = evaluate_offside_position(
        attacker,
        opponents,
        ball_x_m=99.0,
        direction=AttackDirection.RIGHT,
        uncertainty_m=0.25,
    )

    assert decision.status is DecisionStatus.ONSIDE
    assert decision.offside_line_x_m == 99.0
    assert decision.margin_m == -1.0


def test_left_attacking_direction_reverses_ordering() -> None:
    attacker = _player(7, TeamRole.TEAM_A, 18.0)
    opponents = [
        _player(20, TeamRole.TEAM_B, 10.0),
        _player(21, TeamRole.TEAM_B, 20.0),
        _player(22, TeamRole.TEAM_B, 30.0),
    ]

    decision = evaluate_offside_position(
        attacker,
        opponents,
        ball_x_m=25.0,
        direction=AttackDirection.LEFT,
        uncertainty_m=0.25,
    )

    assert decision.status is DecisionStatus.OFFSIDE
    assert decision.second_last_opponent_id == 21
    assert decision.offside_line_x_m == 19.8
    assert decision.margin_m == 2.0


def test_attacker_in_own_half_is_onside() -> None:
    decision = evaluate_offside_position(
        _player(7, TeamRole.TEAM_A, 50.0),
        [
            _player(20, TeamRole.TEAM_B, 48.0),
            _player(21, TeamRole.TEAM_B, 45.0),
        ],
        ball_x_m=40.0,
        direction=AttackDirection.RIGHT,
        uncertainty_m=0.25,
    )

    assert decision.status is DecisionStatus.ONSIDE
    assert "not in the opponents' half" in decision.reason


def test_close_margin_is_uncertain_and_missing_ball_is_unavailable() -> None:
    attacker = _player(7, TeamRole.TEAM_A, 95.2)
    opponents = [
        _player(20, TeamRole.TEAM_B, 100.0),
        _player(21, TeamRole.TEAM_B, 95.0),
    ]

    close = evaluate_offside_position(
        attacker,
        opponents,
        ball_x_m=90.0,
        direction=AttackDirection.RIGHT,
        uncertainty_m=0.50,
    )
    missing = evaluate_offside_position(
        attacker,
        opponents,
        ball_x_m=None,
        direction=AttackDirection.RIGHT,
    )

    assert close.status is DecisionStatus.UNCERTAIN
    assert missing.status is DecisionStatus.UNAVAILABLE


def test_elbows_and_wrists_are_excluded_from_projected_reference() -> None:
    xy = np.column_stack(
        (
            np.linspace(20.0, 36.0, 17),
            np.linspace(10.0, 26.0, 17),
        )
    )
    xy[list(EXCLUDED_ARM_INDICES), 0] = 100.0
    pose = PlayerPose(
        tracker_id=5,
        box_xyxy=np.array([0.0, 0.0, 110.0, 40.0]),
        keypoints_xy=xy,
        confidence=np.ones(17, dtype=np.float64),
    )

    projected = project_player_pose(
        pose,
        TeamRole.TEAM_A,
        np.eye(3),
        DEFAULT_CONFIG,
    )

    assert projected is not None
    assert projected.reference_x(AttackDirection.RIGHT) < 100.0
    assert not pose.legal_mask(0.25)[list(EXCLUDED_ARM_INDICES)].any()
