"""Held-ball displacement rotation dodge — runs ONCE per tick.

Approach: simulate where the carrier would end up after this tick's movement
(me + dir_to_goal * step). If that end position falls inside any opponent's
interaction radius circle, rotate the displacement vector so the tip lands
on the circle boundary (r_int + tolerance). This is a circle-circle
intersection: the carrier moves on a circle of radius `step` around `me`,
and we want the point on that circle at distance `r_eff` from the opponent.
Pick the intersection closest to the desired goal direction.

For multiple opponents, chain: adjust for opp1, then check opp2 against the
adjusted position, etc. A second tick of foresight re-checks from the
predicted end state: if the attacker cannot walk toward the enemy goal now
OR after this tick, the caller must pass before AT shoves them backward.
"""
from __future__ import annotations

import math

from titanium._env import *  # noqa: F401,F403
from titanium.constants import (
    AT_NO_FORWARD_PENALTY,
    FIXED_DT,
    HOLD_OFFSET,
    KEEP_BALL_DANGER_PENALTY,
    WALK_SPEED,
)
from titanium.geometry import rotate_xz, unit_or_zero
from titanium import positioning

# Tolerance added to interaction radius so we don't sit exactly at the limit.
AT_TOLERANCE = 0.1
_PROBE_OFFSETS = (
    Float(0),
    Float(-math.pi / 2),
    Float(math.pi / 2),
    Float(-math.pi / 4),
    Float(math.pi / 4),
)
_UNSAFE_EVAL = Float(-1000)
_SCORE_EVAL = Float(1e6)
_NO_FORWARD_PEN = Float(AT_NO_FORWARD_PENALTY)
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


@cache
def _rotate_away_from_opp(me, end_pos, opp, r_eff, attack_dir=None):
    """If `end_pos` is inside `opp`'s radius `r_eff`, rotate the displacement
    vector (end_pos - me) so the tip lands on the circle boundary.

    Circle-circle intersection: carrier moves on a circle of radius `step`
    around `me`; we want the point on that circle at distance `r_eff` from
    `opp`. Pick the intersection with more forward progress toward
    `attack_dir` (away from own goal). If `attack_dir` is None, fall back to
    closest to original end_pos.

    Returns the adjusted end position (unchanged if not inside the circle).
    """
    displacement = end_pos - me
    step_len = Magnitude(displacement)

    opp_offset = opp - me
    d = Magnitude(opp_offset)
    d_safe = ConditionalSetFloat(CompareFloats(d, Float(1e-4), "<"), Float(1e-4), d)

    # Distance from end_pos to opp
    end_to_opp = Distance(end_pos, opp)
    inside = CompareFloats(end_to_opp, r_eff, "<=")

    # Circle-circle intersection:
    #   a = (step^2 - r_eff^2 + d^2) / (2*d)
    #   h = sqrt(step^2 - a^2)
    #   midpoint = me + (opp - me) * (a/d)
    #   perp = perpendicular to (opp - me) in XZ
    #   point = midpoint +/- perp * h
    a = (MultiplyFloats(step_len, step_len) - MultiplyFloats(r_eff, r_eff) + MultiplyFloats(d, d)) / (Float(2.0) * d_safe)
    a_clamped = ClampFloat(a, Float(0), step_len)
    h_sq = MultiplyFloats(step_len, step_len) - MultiplyFloats(a_clamped, a_clamped)
    h_sq = ConditionalSetFloat(CompareFloats(h_sq, Float(0), "<"), Float(0), h_sq)
    h = Sqrt(h_sq)

    toward_opp = unit_or_zero(opp_offset)
    parts = Vector3Split(toward_opp)
    # Perpendicular in XZ: (z, 0, -x)
    perp = Vector3(parts.z, Float(0), Float(0) - parts.x)

    midpoint = me + toward_opp * a_clamped
    offset = perp * h

    # Two intersection candidates
    cand_a = midpoint + offset
    cand_b = midpoint - offset

    if attack_dir is not None:
        # Prefer the candidate with more forward progress (away from own goal).
        # Walking into own goal is just as bad as getting tackled.
        prog_a = DotProduct(unit_or_zero(cand_a - me), attack_dir)
        prog_b = DotProduct(unit_or_zero(cand_b - me), attack_dir)
        prefer_a = CompareFloats(prog_a, prog_b, ">=")
    else:
        # Fallback: pick the one closer to the original end_pos
        dist_a = Distance(cand_a, end_pos)
        dist_b = Distance(cand_b, end_pos)
        prefer_a = CompareFloats(dist_a, dist_b, "<=")

    rotated = ConditionalSetVector3(prefer_a, cand_a, cand_b)

    # Only apply rotation when inside; otherwise keep original
    return ConditionalSetVector3(inside, rotated, end_pos)


@cache
def _rotate_away_from_opp_dir(me, direction, step_len, opp, r_eff):
    """Same as `_rotate_away_from_opp` but works from a direction + step length.
    Returns the adjusted direction (unit vector) and whether rotation was applied.
    """
    end_pos = me + direction * step_len
    adjusted = _rotate_away_from_opp(me, end_pos, opp, r_eff)
    adjusted_dir = unit_or_zero(adjusted - me)
    end_to_opp = Distance(end_pos, opp)
    inside = CompareFloats(end_to_opp, r_eff, "<=")
    return adjusted_dir, inside


def _displacement_walk(me, desired_dir, opponents, staminas, r_int, my_stam, step_len, attack_dir=None):
    """Compute walk direction by simulating movement and rotating away from
    any opponent whose interaction circle the end position falls inside.

    Chains over opponents: adjust for opp1, then check opp2 against adjusted
    position, etc. Only stam-capable opponents steer the walk.

    If `attack_dir` is provided, rotations bias toward forward progress —
    walking into your own goal is treated as equally bad as getting tackled.
    After all rotations, if the final direction retreats (negative dot with
    attack_dir) but the original was forward, revert to the original direction.
    """
    r_eff = r_int + Float(AT_TOLERANCE)
    original_end = me + desired_dir * step_len
    end_pos = original_end

    any_adjusted = Bool(False)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        # Predicted opponent position after this tick (they step toward us)
        opp_end = step_toward(opp, me, _WALK, _DT)
        end_to_opp = Distance(end_pos, opp_end)
        inside = And(capable, CompareFloats(end_to_opp, r_eff, "<="))
        adjusted = _rotate_away_from_opp(me, end_pos, opp_end, r_eff, attack_dir)
        end_pos = ConditionalSetVector3(inside, adjusted, end_pos)
        any_adjusted = Or(any_adjusted, inside)

    final_dir = unit_or_zero(end_pos - me)

    # Own-goal guard: if the rotated direction retreats but the original was
    # forward, getting tackled is better than walking into our own goal.
    # Revert to the original desired direction.
    if attack_dir is not None:
        original_forward = CompareFloats(DotProduct(desired_dir, attack_dir), Float(0), ">=")
        final_retreats = CompareFloats(DotProduct(final_dir, attack_dir), Float(0), "<")
        revert = And(any_adjusted, And(original_forward, final_retreats))
        final_dir = ConditionalSetVector3(revert, desired_dir, final_dir)
        end_pos = ConditionalSetVector3(revert, original_end, end_pos)

    return final_dir, any_adjusted


def _any_forward_walk(me, desired_dir, opponents, staminas, r_int, my_stam, step_len, attack_dir=None):
    """True if the displacement-rotated walk still makes forward progress."""
    final_dir, _ = _displacement_walk(me, desired_dir, opponents, staminas, r_int, my_stam, step_len, attack_dir)
    progress = DotProduct(final_dir, desired_dir)
    return CompareFloats(progress, Float(0), ">=")


def aim_is_safe(me, opponents, r_int, my_stam, hold=HOLD_OFFSET):
    """Build a predicate: can the carrier aim `direction` and keep the ball?

    Uses the displacement rotation model: simulate where the held ball would
    land (me_end + direction * hold), and check if any stam-capable opponent
    can reach it. If the ball position falls inside an opponent's circle,
    it's not safe to aim that way.
    """
    staminas = opponent_staminas()
    r_eff = r_int + Float(AT_TOLERANCE)

    def safe(direction):
        # Held ball position: carrier end + aim direction * hold
        _me_end, ball_end = simulate_end_of_tick(me, direction, 5.5, hold)
        blocked = Bool(False)
        for opp, stam in zip(opponents, staminas):
            capable = _can_tackle_us(stam, my_stam)
            opp_end = step_toward(opp, me, _WALK, _DT)
            d = Distance(ball_end, opp_end)
            in_range = And(capable, CompareFloats(d, r_eff, "<="))
            blocked = Or(blocked, in_range)
        return Not(blocked)

    return safe


def heading_safe(
    me,
    heading,
    ball_start,
    opponents,
    staminas,
    r_int,
    team_goal,
    my_stam,
    opp_goal,
    hold=HOLD_OFFSET,
    move_step=5.5,
):
    """Can the ball be HELD along `heading` for one tick without being taken?"""
    _me_end, ball_end = simulate_end_of_tick(me, heading, move_step, hold)
    r_eff = r_int + Float(AT_TOLERANCE)

    threat = Bool(False)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        opp_end = step_toward(opp, me, _WALK, _DT)
        d = Distance(ball_end, opp_end)
        in_range = And(capable, CompareFloats(d, r_eff, "<="))
        threat = Or(threat, in_range)

    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    tackle_threat = And(threat, Not(enemy_net))
    return Or(enemy_net, And(Not(tackle_threat), own_goal_ok))


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
    """Probe a single heading offset for safety, eval, and support outlets."""
    heading = rotate_xz(desired_dir, offset)
    me_end, ball_end = simulate_end_of_tick(me, heading, move_step, hold)
    safe = heading_safe(
        me, heading, ball_start, opponents, staminas, r_int, team_goal, my_stam,
        opp_goal, hold, move_step,
    )
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    progress = DotProduct(heading, unit_or_zero(opp_goal - me))
    forward = CompareFloats(progress, Float(0), ">=")
    usable = Or(enemy_net, And(safe, forward))
    # Clearance: min distance to any stam-capable opponent
    r_eff = r_int + Float(AT_TOLERANCE)
    clearance = Float(1e6)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        opp_end = step_toward(opp, me, _WALK, _DT)
        d = Distance(opp_end, ball_end)
        d_eff = ConditionalSetFloat(capable, d, Float(1e6))
        clearance = ConditionalSetFloat(CompareFloats(d_eff, clearance, "<"), d_eff, clearance)
    keep_pen = ConditionalSetFloat(danger, Float(KEEP_BALL_DANGER_PENALTY), Float(0))
    raw_eval = progress * Float(2) + clearance * Float(0.25) - keep_pen
    raw_eval = ConditionalSetFloat(enemy_net, _SCORE_EVAL, raw_eval)
    eval_score = ConditionalSetFloat(usable, raw_eval, _UNSAFE_EVAL)
    worst_stam = Float(-1)
    tackleable = Bool(False)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        opp_end = step_toward(opp, me, _WALK, _DT)
        d = Distance(opp_end, ball_end)
        in_range = And(capable, CompareFloats(d, r_eff, "<="))
        tackleable = Or(tackleable, in_range)
        worst_stam = ConditionalSetFloat(
            And(in_range, CompareFloats(stam, worst_stam, ">")), stam, worst_stam
        )
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
    """Displacement-rotation anti-tackle: simulate the movement vector, and if
    its tip lands inside any opponent's interaction circle, rotate the vector
    so the tip sits on the circle boundary (r_int + tolerance). Pick the
    intersection closest to the desired goal direction.

    Tick+1 foresight: from the predicted end state, re-check forward progress.
    No forward walk now or next => need_pass (pass before AT shoves us back).
    """
    if danger is None:
        danger = Bool(False)
    desired_dir = unit_or_zero(desired - me)
    staminas = opponent_staminas()
    opp_goal = desired
    step_len = Float(move_step)

    # Attack direction for own-goal-aware rotation
    attack_dir = unit_or_zero(opp_goal - me)

    # Core: displacement rotation away from opponents (bias toward attack dir)
    best_dir, any_adjusted = _displacement_walk(
        me, desired_dir, opponents, staminas, r_int, my_stam, step_len, attack_dir
    )
    # If no adjustment was needed, best_dir is the same as desired_dir
    any_open = Not(any_adjusted)
    # When adjusted, the rotated direction is still usable (by construction it
    # sits on the circle boundary, so the opponent can't reach it)
    any_open = Or(any_open, any_adjusted)

    any_forward = _any_forward_walk(
        me, desired_dir, opponents, staminas, r_int, my_stam, step_len, attack_dir
    )

    # Probes for support outlets (five fixed offsets, same as before)
    probes = [
        _probe(
            me, desired_dir, off, ball_start, opponents, staminas,
            r_int, hold, move_step, team_goal, my_stam, opp_goal, danger,
        )
        for off in _PROBE_OFFSETS
    ]

    # Own-goal guard
    me_end, ball_end = simulate_end_of_tick(me, best_dir, move_step, hold)
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    predicted_safe = Or(enemy_net, And(any_open, own_goal_ok))
    toward_attack = unit_or_zero(opp_goal - me)
    retreating = CompareFloats(DotProduct(best_dir, toward_attack), Float(0), "<")
    own_goal_push = And(any_open, Not(own_goal_ok))

    # Tick+1 foresight: from predicted end state, check forward progress
    opp_ends = [step_toward(opp, me, _WALK, _DT) for opp in opponents]
    desired_next = unit_or_zero(opp_goal - me_end)
    attack_dir_next = unit_or_zero(opp_goal - me_end)
    any_forward_next = _any_forward_walk(
        me_end, desired_next, opp_ends, staminas, r_int, my_stam, step_len, attack_dir_next
    )

    trapped_now = Not(any_forward)
    trapped_next = Not(any_forward_next)
    no_forward = Or(trapped_now, trapped_next)
    need_pass = Or(no_forward, own_goal_push)

    best_eval = DotProduct(best_dir, desired_dir) - ConditionalSetFloat(
        no_forward, _NO_FORWARD_PEN, Float(0)
    )

    return best_dir, {
        "any_safe": any_open,
        "any_forward": any_forward,
        "any_forward_next": any_forward_next,
        "predicted_safe": predicted_safe,
        "probes": probes,
        "chosen_offset": Float(0),
        "best_eval": best_eval,
        "need_pass": need_pass,
        "retreating": Or(retreating, no_forward),
        "own_goal_push": own_goal_push,
        "trapped_next": trapped_next,
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
    walk = me + direction * Float(step)
    walk = ConditionalSetVector3(debug["own_goal_push"], me, walk)
    return walk, debug
