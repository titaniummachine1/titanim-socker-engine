"""Shared end-of-tick positioning rules (same model as anti_tackle).

All placement / dodge decisions that care about "where will bodies be after
this tick" must use these helpers — never live positions alone.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import (
    FIXED_DT,
    GOAL_HALF_WIDTH,
    HOLD_OFFSET,
    OWN_GOAL_PLANE_CLEARANCE,
    PASS_DANGER_APPROACH_FRAC,
    PASS_DANGER_TRACK_MULT,
    PITCH_X_MAX,
    PITCH_X_MIN,
    PITCH_Z_MAX,
    PITCH_Z_MIN,
    WALK_SPEED,
)
from titanium.geometry import unit_or_zero
from titanium.nodefn import graph_function

_WALK = Float(WALK_SPEED)
_DT = Float(FIXED_DT)
_HOLD = Float(HOLD_OFFSET)
_CLEAR = Float(OWN_GOAL_PLANE_CLEARANCE)
_XMIN = Float(PITCH_X_MIN)
_XMAX = Float(PITCH_X_MAX)
_ZMIN = Float(PITCH_Z_MIN)
_ZMAX = Float(PITCH_Z_MAX)
_MOUTH = Float(GOAL_HALF_WIDTH)


@cache
def step_toward(pos, target, speed, dt):
    direction = unit_or_zero(target - pos)
    return pos + direction * speed * dt


def _clamp_axis(v, lo, hi):
    return ClampFloat(v, lo, hi)


@cache
def clamp_hold_to_playable(hold):  # inlined: 100 evals vs 140 as a function
    """Project held-ball center into playable AABB — matches sim
    `project_hold_into_playable`: Z always clamped; X clamped only outside
    the goal mouth so walk-in goals stay possible and wall-rub compresses
    the hold offset exactly like Unity.
    """
    z = _clamp_axis(hold.z, _ZMIN, _ZMAX)
    in_mouth = CompareFloats(Abs(z), _MOUTH, "<=")
    x_clamped = _clamp_axis(hold.x, _XMIN, _XMAX)
    x = ConditionalSetFloat(in_mouth, hold.x, x_clamped)
    return Vector3(x, hold.y, z)


def simulate_end_of_tick(me, heading, move_step, hold=_HOLD):
    """Carrier + held-ball positions after one walk tick along MoveTo.

    Ball uses the ideal hold offset then is clamped to pitch bounds — same
    as the simulator's hold_pos_playable — so AT near walls sees the real
    compressed ball position, not a phantom off-pitch point.
    """
    move_to = me + heading * Float(move_step)
    me_end = step_toward(me, move_to, _WALK, _DT)
    ball_raw = me_end + heading * Float(hold)
    ball_end = clamp_hold_to_playable(ball_raw)
    return me_end, ball_end


def simulate_walk_end(me, move_target):
    """Where `me` ends if MoveTo = `move_target` this tick (no held ball)."""
    return step_toward(me, move_target, _WALK, _DT)


@cache
def ball_clear_of_own_goal_plane(ball_end, team_goal, clearance=_CLEAR):  # inlined: 61 vs 160
    """True if held ball stays on the pitch side of our goal plane.

    Own-goal risk: anti-tackle rotates the hold offset and can put the ball
    across the line. Clearance includes hold length + ball radius + margin.
    """
    defends_neg = CompareFloats(team_goal.x, Float(0), "<")
    ok_west = CompareFloats(ball_end.x, team_goal.x + clearance, ">=")
    ok_east = CompareFloats(ball_end.x, team_goal.x - clearance, "<=")
    return ConditionalSetBool(defends_neg, ok_west, ok_east)


@cache
def ball_in_goal_net(ball_end, goal):  # inlined: 224 evals vs 360 as a function
    """True if ball center is past `goal`'s line inside the mouth (sim `goal_at`).

    Held or free — once the ball is in the net it is a goal; it cannot be
    tackled. `goal` is Team/Opponent Goal Center (x marks the goal line).
    """
    in_mouth = CompareFloats(Abs(ball_end.z), _MOUTH, "<=")
    east_net = CompareFloats(goal.x, Float(0), ">")
    past_east = CompareFloats(ball_end.x, goal.x, ">=")
    past_west = CompareFloats(ball_end.x, goal.x, "<=")
    past = ConditionalSetBool(east_net, past_east, past_west)
    return And(past, in_mouth)


def pass_danger_radius(r_int):
    """Opp closer than this to carrier center → seek pass (touch is too late).

    Track bubble = 2×r_int. Closed more than 1/4 of that approach ⇒ remaining
    gap ≤ ¾·2r = 1.5·r_int.
    """
    return r_int * Float(PASS_DANGER_TRACK_MULT * (1.0 - PASS_DANGER_APPROACH_FRAC))


def opponent_in_pass_danger(me, opponents, r_int, my_stam=None, opp_staminas=None):
    """Pass-danger only from opps who can win a tackle (stam >= ours).

    Exhausted defenders are ghosts for dribble/AT — walk through them.
    """
    danger_r = pass_danger_radius(r_int)
    flagged = Bool(False)
    stams = opp_staminas
    if stams is None:
        stams = [SoccerGetFloat(f"Opponent Player {i} Stamina") for i in range(1, 5)]
    my = my_stam if my_stam is not None else Float(0)
    for opp, stam in zip(opponents, stams):
        capable = CompareFloats(stam, my, ">=")
        opp_end = step_toward(opp, me, _WALK, _DT)
        near = CompareFloats(Distance(opp_end, me), danger_r, "<=")
        flagged = Or(flagged, And(capable, near))
    return flagged


def preferred_spacing(r_int):
    """Formation heuristic gap between role stations (not a physics constraint).

    Uses r_int + HOLD_OFFSET as a convenient pitch-scale preferred distance —
    historically the held-ball steal radius; here it only means "don't park
    two roles on top of each other."
    """
    return r_int + Float(HOLD_OFFSET)


def reduce_role_overlap(anchor, point, spacing, strength=None):
    """Nudge `point` partway toward the preferred ring around `anchor`.

    Applies `strength * (spacing - dist)` outward — not a hard snap to the
    ring — so two near-equal roles do not bounce each tick at 1 FPS.
    """
    from titanium.constants import OVERLAP_SHIFT_STRENGTH

    if strength is None:
        strength = Float(OVERLAP_SHIFT_STRENGTH)
    delta = point - anchor
    dist = Magnitude(delta)
    too_close = CompareFloats(dist, spacing, "<")
    safe_dir = unit_or_zero(
        ConditionalSetVector3(
            CompareFloats(dist, Float(1e-4), "<"),
            Vector3(Float(1), Float(0), Float(0)),
            delta,
        )
    )
    gap = spacing - dist
    nudged = point + safe_dir * gap * strength
    return ConditionalSetVector3(too_close, nudged, point)


# Back-compat aliases (old collision-flavoured names).
teammate_min_separation = preferred_spacing
push_apart_from = reduce_role_overlap


def task_value(
    has_ball,
    team_has,
    opp_has,
    loose,
    closest_to_ball,
    duty,
    is_press_body,
    emergency=None,
):
    """Expected tactical value of this player's current task (higher keeps space).

    emergency (goal prevention / net-bound claim) outranks carrier movement.
    """
    from titanium.constants import (
        PRI_CARRIER,
        PRI_EMERGENCY,
        PRI_LOOSE_CLAIM,
        PRI_PRESS_TACKLE,
        PRI_SUPPORT,
        PRI_THREAT_COVER,
    )

    if emergency is None:
        emergency = Bool(False)

    # Low → high so later overrides win.
    p = Float(PRI_THREAT_COVER)
    p = ConditionalSetFloat(And(team_has, Not(has_ball)), Float(PRI_SUPPORT), p)
    p = ConditionalSetFloat(has_ball, Float(PRI_CARRIER), p)
    p = ConditionalSetFloat(And(loose, closest_to_ball), Float(PRI_LOOSE_CLAIM), p)
    p = ConditionalSetFloat(And(opp_has, Or(duty, is_press_body)), Float(PRI_PRESS_TACKLE), p)
    p = ConditionalSetFloat(emergency, Float(PRI_EMERGENCY), p)
    return p


movement_priority = task_value  # alias


def resolve_space_claims(
    me,
    desired_target,
    peer_bodies,
    peer_values,
    my_value,
    r_int,
    ball=None,
    opp_goal=None,
    carrier_retreating=None,
    collapse_ok=None,
):
    """Soft utility resolver for overlapping role stations.

    Resolve only against the strongest conflicting claim (value gap + overlap)
    so teammate list order cannot steer the result.
    """
    from titanium.constants import (
        COLLAPSE_SPACING_FRAC,
        MILD_SPREAD_FRAC,
        OVERLAP_SHIFT_STRENGTH,
        PRI_CARRIER,
        PRI_VALUE_BAND,
        ROLE_SHIFT_FRAC,
    )

    pref = preferred_spacing(r_int)
    mild = pref * Float(MILD_SPREAD_FRAC)
    shift = pref * Float(ROLE_SHIFT_FRAC)
    collapse = pref * Float(COLLAPSE_SPACING_FRAC)
    band = Float(PRI_VALUE_BAND)
    carrier_v = Float(PRI_CARRIER)
    strength = Float(OVERLAP_SHIFT_STRENGTH)
    if collapse_ok is None:
        collapse_ok = Bool(False)

    me_end0 = simulate_walk_end(me, desired_target)

    # Pick the single strongest conflict: (their_v - my_v) + positive overlap.
    best_peer = peer_bodies[0]
    best_v = peer_values[0]
    d0 = Distance(me_end0, best_peer)
    ov0 = pref - d0
    ov0_pos = ConditionalSetFloat(CompareFloats(ov0, Float(0), ">"), ov0, Float(0))
    best_score = (best_v - my_value) + ov0_pos

    for peer, their_v in zip(peer_bodies[1:], peer_values[1:]):
        d = Distance(me_end0, peer)
        ov = pref - d
        ov_pos = ConditionalSetFloat(CompareFloats(ov, Float(0), ">"), ov, Float(0))
        score = (their_v - my_value) + ov_pos
        better = CompareFloats(score, best_score, ">")
        best_peer = ConditionalSetVector3(better, peer, best_peer)
        best_v = ConditionalSetFloat(better, their_v, best_v)
        best_score = ConditionalSetFloat(better, score, best_score)

    me_end = me_end0
    lower = CompareFloats(my_value + band, best_v, "<")
    near_eq = CompareFloats(Abs(my_value - best_v), band, "<=")
    vs_carrier = CompareFloats(Abs(best_v - carrier_v), band, "<=")
    lower_gap = ConditionalSetFloat(And(collapse_ok, vs_carrier), collapse, shift)

    after_eq = ConditionalSetVector3(
        And(near_eq, CompareFloats(Distance(me_end, best_peer), mild, "<")),
        reduce_role_overlap(best_peer, me_end, mild, strength),
        desired_target,
    )
    # Large value gap: full-strength shift; near-equal already used soft strength.
    target = ConditionalSetVector3(
        And(lower, CompareFloats(Distance(me_end, best_peer), lower_gap, "<")),
        reduce_role_overlap(best_peer, me_end, lower_gap, Float(1.0)),
        after_eq,
    )

    if ball is not None and opp_goal is not None and carrier_retreating is not None:
        attack_east = CompareFloats(opp_goal.x, Float(0), ">")
        past_east = CompareFloats(target.x, ball.x, ">")
        past_west = CompareFloats(target.x, ball.x, "<")
        past_ball = ConditionalSetBool(attack_east, past_east, past_west)
        capped = Vector3(ball.x, target.y, target.z)
        target = ConditionalSetVector3(And(carrier_retreating, past_ball), capped, target)
    return target


def resolve_space_claims_legacy(
    me,
    desired_target,
    anchors,
    r_int,
    priority_anchor=None,
    ball=None,
    opp_goal=None,
    carrier_retreating=None,
):
    """Legacy: treat every anchor as higher-value; shift off all of them."""
    pref = preferred_spacing(r_int)
    target = desired_target
    ordered = list(anchors)
    if priority_anchor is not None:
        ordered = [priority_anchor] + ordered
    for anchor in ordered:
        me_end = simulate_walk_end(me, target)
        need = CompareFloats(Distance(me_end, anchor), pref, "<")
        shifted = reduce_role_overlap(anchor, me_end, pref)
        target = ConditionalSetVector3(need, shifted, target)

    if ball is not None and opp_goal is not None and carrier_retreating is not None:
        attack_east = CompareFloats(opp_goal.x, Float(0), ">")
        past_east = CompareFloats(target.x, ball.x, ">")
        past_west = CompareFloats(target.x, ball.x, "<")
        past_ball = ConditionalSetBool(attack_east, past_east, past_west)
        capped = Vector3(ball.x, target.y, target.z)
        target = ConditionalSetVector3(And(carrier_retreating, past_ball), capped, target)
    return target


def clamp_support_station(
    me,
    support_target,
    anchors,
    r_int,
    priority_anchor=None,
    ball=None,
    opp_goal=None,
    carrier_retreating=None,
    my_priority=None,
    peer_priorities=None,
    collapse_ok=None,
):
    """Compatibility wrapper → resolve_space_claims / legacy."""
    if my_priority is not None and peer_priorities is not None:
        return resolve_space_claims(
            me,
            support_target,
            anchors,
            peer_priorities,
            my_priority,
            r_int,
            ball=ball,
            opp_goal=opp_goal,
            carrier_retreating=carrier_retreating,
            collapse_ok=collapse_ok,
        )
    return resolve_space_claims_legacy(
        me,
        support_target,
        anchors,
        r_int,
        priority_anchor=priority_anchor,
        ball=ball,
        opp_goal=opp_goal,
        carrier_retreating=carrier_retreating,
    )
