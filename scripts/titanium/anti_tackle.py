"""Held-ball rotation dodge — runs ONCE per tick.

Probes headings; marks which end-of-tick ball positions would be tackled
(stam-capable opps only). Among SAFE probes, pick the heading that
maximizes walk eval (progress + corridor + clearance). If no safe heading,
caller should pass to a teammate.
"""
from __future__ import annotations

import math

from titanium._env import *  # noqa: F401,F403
from titanium.constants import FIXED_DT, HOLD_OFFSET, KEEP_BALL_DANGER_PENALTY, WALK_SPEED
from titanium.geometry import rotate_xz, unit_or_zero
from titanium import positioning

_PROBE_OFFSETS = (
    Float(0),
    Float(-math.pi / 2),
    Float(math.pi / 2),
    Float(-math.pi / 4),
    Float(math.pi / 4),
)
_UNSAFE_EVAL = Float(-1000)
_SCORE_EVAL = Float(1e6)
_WALK = Float(WALK_SPEED)
_DT = Float(FIXED_DT)

step_toward = positioning.step_toward
simulate_end_of_tick = positioning.simulate_end_of_tick


def _opp_stamina(slot: int):
    return SoccerGetFloat(f"Opponent Player {slot} Stamina")


def opponent_staminas():
    return [_opp_stamina(i) for i in range(1, 5)]


def _can_tackle_us(opp_stam, my_stam):
    """Tackle duel: only stam >= carrier can dispossess. Lower stam = ghost for AT."""
    return CompareFloats(opp_stam, my_stam, ">=")


def _opponent_near_carrier(opp, ball_start, me, r_int):
    two_r = r_int * Float(2)
    near_me = CompareFloats(Distance(me, opp), two_r, "<=")
    near_ball = CompareFloats(Distance(ball_start, opp), two_r, "<=")
    return Or(near_me, near_ball)


def _end_tick_tackleable(ball_end, ball_start, opponents, staminas, r_int, me, my_stam):
    """Low-stam opps are non-threats — walk straight through them (AT ignores)."""
    threat = Bool(False)
    worst_stam = Float(-1)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        near_zone = _opponent_near_carrier(opp, ball_start, me, r_int)
        # Ghosts: not capable → never threaten, never affect AT.
        tracks = And(capable, near_zone)
        opp_end = step_toward(opp, ball_start, _WALK, _DT)
        near = CompareFloats(Distance(opp_end, ball_end), r_int, "<=")
        threatens = And(tracks, near)
        threat = Or(threat, threatens)
        worst_stam = ConditionalSetFloat(
            And(threatens, CompareFloats(stam, worst_stam, ">")),
            stam,
            worst_stam,
        )
    return threat, worst_stam


def _min_stam_capable_clearance(ball_end, ball_start, opponents, staminas, me, my_stam, r_int):
    """Clearance only vs stam-capable threats — ghosts do not steer the walk."""
    best = Float(1e6)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        near_zone = _opponent_near_carrier(opp, ball_start, me, r_int)
        tracks = And(capable, near_zone)
        opp_end = step_toward(opp, ball_start, _WALK, _DT)
        d = Distance(opp_end, ball_end)
        d_eff = ConditionalSetFloat(tracks, d, Float(1e6))
        best = ConditionalSetFloat(CompareFloats(d_eff, best, "<"), d_eff, best)
    return best


def probe_walk_eval(me, me_end, ball_end, heading, opp_goal, clearance, danger):
    """Maximize: progress to goal + face-goal geometry + clearance − danger keep penalty."""
    to_goal = unit_or_zero(opp_goal - me)
    progress = DotProduct(heading, to_goal)  # [-1,1] toward goal
    closer = Distance(me, opp_goal) - Distance(me_end, opp_goal)
    face = DotProduct(unit_or_zero(ball_end - me_end), unit_or_zero(opp_goal - me_end))
    margin = clearance
    keep_pen = ConditionalSetFloat(danger, Float(KEEP_BALL_DANGER_PENALTY), Float(0))
    return progress * Float(2) + closer * Float(0.5) + face + margin * Float(0.25) - keep_pen


def _probe(
    me,
    desired_dir,
    offset,
    ball_start,
    opponents,
    staminas,
    r_int,
    hold,
    move_step,
    team_goal,
    my_stam,
    opp_goal,
    danger,
):
    heading = rotate_xz(desired_dir, offset)
    me_end, ball_end = simulate_end_of_tick(me, heading, move_step, hold)
    tackleable, worst_stam = _end_tick_tackleable(
        ball_end, ball_start, opponents, staminas, r_int, me, my_stam
    )
    # Enemy net = goal already. Ball in goal cannot be tackled — force safe.
    # Own net = never safe (void any "place it" temptation).
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    tackle_threat = And(tackleable, Not(enemy_net))
    safe = Or(enemy_net, And(Not(tackle_threat), own_goal_ok))
    # Forward = eval-up toward goal. Scoring into their net is always usable.
    progress = DotProduct(heading, unit_or_zero(opp_goal - me))
    forward = CompareFloats(progress, Float(0), ">=")
    usable = Or(enemy_net, And(safe, forward))
    clearance = _min_stam_capable_clearance(
        ball_end, ball_start, opponents, staminas, me, my_stam, r_int
    )
    raw_eval = probe_walk_eval(me, me_end, ball_end, heading, opp_goal, clearance, danger)
    # Prefer placing the ball in their net over every ordinary walk eval.
    raw_eval = ConditionalSetFloat(enemy_net, _SCORE_EVAL, raw_eval)
    eval_score = ConditionalSetFloat(usable, raw_eval, _UNSAFE_EVAL)
    return safe, usable, heading, Abs(offset), tackleable, worst_stam, eval_score, progress


def search_safe_direction(
    me,
    desired,
    ball_start,
    opponents,
    r_int,
    team_goal,
    my_stam,
    danger=None,
    hold=HOLD_OFFSET,
    move_step=5.5,
):
    """Among safe+forward probes, pick max eval.

    If the only safe options go backward (hurt eval), need_pass — do not retreat.
    """
    if danger is None:
        danger = Bool(False)
    desired_dir = unit_or_zero(desired - me)
    staminas = opponent_staminas()
    opp_goal = desired

    probes = [
        _probe(
            me,
            desired_dir,
            off,
            ball_start,
            opponents,
            staminas,
            r_int,
            hold,
            move_step,
            team_goal,
            my_stam,
            opp_goal,
            danger,
        )
        for off in _PROBE_OFFSETS
    ]

    best_dir = desired_dir
    best_eval = _UNSAFE_EVAL
    best_offset = Float(0)
    any_usable = Bool(False)
    any_safe = Bool(False)

    for off, (safe, usable, direction, abs_off, _tackleable, _worst, eval_score, _prog) in zip(
        _PROBE_OFFSETS, probes
    ):
        any_safe = Or(any_safe, safe)
        better = And(usable, Or(Not(any_usable), CompareFloats(eval_score, best_eval, ">")))
        best_dir = ConditionalSetVector3(better, direction, best_dir)
        best_eval = ConditionalSetFloat(better, eval_score, best_eval)
        best_offset = ConditionalSetFloat(better, off, best_offset)
        any_usable = Or(any_usable, usable)

    _me_end, ball_end = simulate_end_of_tick(me, best_dir, move_step, hold)
    predicted_tackleable, _ = _end_tick_tackleable(
        ball_end, ball_start, opponents, opponent_staminas(), r_int, me, my_stam
    )
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    predicted_safe = Or(
        enemy_net,
        And(Not(And(predicted_tackleable, Not(enemy_net))), own_goal_ok),
    )
    # No forward-safe walk (only back / none) → pass, never walk backward.
    # Exception: enemy-net finish is usable even if classified oddly.
    need_pass = Not(any_usable)

    return best_dir, {
        "any_safe": any_safe,
        "any_forward": any_usable,
        "predicted_safe": predicted_safe,
        "probes": probes,
        "chosen_offset": best_offset,
        "best_eval": best_eval,
        "need_pass": need_pass,
    }


def carrier_walk_target(
    me,
    desired,
    ball_start,
    opponents,
    r_int,
    team_goal,
    my_stam,
    danger=None,
    step=5.5,
    hold=HOLD_OFFSET,
):
    direction, debug = search_safe_direction(
        me, desired, ball_start, opponents, r_int, team_goal, my_stam, danger, hold, step
    )
    return me + direction * Float(step), debug
