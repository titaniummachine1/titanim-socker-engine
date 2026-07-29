"""Shot legality and walking: is there a clean lane right now, and how we
move toward a target (straight line — dodge logic lives in anti_tackle).

Opponent interception circle optimization: for each opponent, compute the
left-most and right-most tangent directions to their interception radius
circle (same tangent-circle construction as post tangents). A shot direction
that falls BETWEEN these two tangent angles is intercepted; a direction
OUTSIDE them is clear of that opponent. These tangent directions are also
used as fallback shot candidates — if the three primary candidates (center,
left post, right post) are all blocked, we try grazing past the opponent
circles, picking the legal direction closest to the goal-center line.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import CARRIER_SPEED, POST_AIM_RADIUS, SPRINT_SPEED
from titanium.geometry import (
    closest_point_on_goal_mouth,
    is_legal_direction,
    opponent_half_angle,
    opponent_invariants,
    opponent_tangents,
    post_tangent,
    unit_or_zero,
)


def walk_target(me, desired, step=5.5):
    """Walk straight toward `desired` — no cone avoidance."""
    direction = unit_or_zero(desired - me)
    return me + direction * Float(step)


def clear_shot(
    shot_origin, opp_goal, opp_left_post, opp_right_post, opponents, r_eff,
    direction_ok=None,
):
    """Is there a legal straight-line lane into the enemy goal right now, and
    along which direction?

    Candidate priority (first legal wins):

    1. **Center** — straight at the goal centre (largest margin for error).
    2. **Left post tangent** — true max-angle edge corrected for post+ball
       radius.
    3. **Right post tangent** — true min-angle edge.
    4. **Opponent tangent fallbacks** — for each opponent, the left and right
       tangent directions to their interception radius circle (same
       tangent-circle construction as posts). These are the directions that
       graze just past the opponent's reach, picked closest to the goal-center
       line among legal ones.

    ALL opponents count here (including exhausted ones): a released ball is
    loose and anyone can pick it up. Anti-tackle/dribble ghosts low-stam
    bodies; shooting must not.

    `direction_ok` is an optional extra test applied per candidate. The
    anti-tackle build passes one: aiming a direction turns the carrier, which
    parks the held ball along that direction, so a lane that is geometrically
    open is still worthless if the ball gets taken the moment you point at it.

    Returns `(lane_open, aim_direction)`.
    """
    dir_c = unit_or_zero(opp_goal - shot_origin)
    dir_l, _ = post_tangent(shot_origin, opp_left_post, POST_AIM_RADIUS)
    dir_r, _ = post_tangent(shot_origin, opp_right_post, POST_AIM_RADIUS)

    half_angles = [opponent_half_angle(o, shot_origin, r_eff) for o in opponents]
    # Skip defenders further from the goal than the ball (plus margin):
    # the ball reaches the goal before they can catch up.  Done inside the
    # graph by OR-ing engulfed with a "too far behind" flag so the cone
    # check never fires for trailing opponents.
    margin = r_eff + Float(0.1)
    d_to_goal = Distance(shot_origin, opp_goal)
    invariants = []
    for o, ha in zip(opponents, half_angles):
        toward, cos_a, engulfed = opponent_invariants(o, shot_origin, r_eff, ha)
        too_far_behind = CompareFloats(Distance(o, opp_goal), d_to_goal + margin, ">")
        # If too far behind, force engulfed=False so cone never blocks.
        engulfed = And(engulfed, Not(too_far_behind))
        # Also widen the cone to nothing by setting cos_a above any dot product.
        cos_a = ConditionalSetFloat(too_far_behind, Float(2.0), cos_a)
        invariants.append((toward, cos_a, engulfed))

    def viable(cand):
        legal = is_legal_direction(cand, invariants)
        return legal if direction_ok is None else And(legal, direction_ok(cand))

    # Primary candidates: Center -> Left Post -> Right Post
    best = dir_c
    ok = viable(dir_c)
    for cand in (dir_l, dir_r):
        legal = viable(cand)
        take = And(legal, Not(ok))
        best = ConditionalSetVector3(take, cand, best)
        ok = Or(ok, legal)

    # Fallback: opponent tangent directions (graze past interception circles)
    # Same logic as post tangents — left/right tangent to each opponent's
    # interception radius circle. Only tried when all 3 primary candidates
    # are blocked. Among legal fallbacks, prefer closest to goal-center line.
    for opp, ha in zip(opponents, half_angles):
        tang_l, tang_r = opponent_tangents(opp, shot_origin, ha)
        for cand in (tang_l, tang_r):
            legal = viable(cand)
            take = And(legal, Not(ok))
            best = ConditionalSetVector3(take, cand, best)
            ok = Or(ok, legal)

    return ok, best


def nearest_opponent_dist(opponents, ball):
    nearest_d = Distance(opponents[0], ball)
    for opp in opponents[1:]:
        d = Distance(opp, ball)
        nearest_d = ConditionalSetFloat(CompareFloats(d, nearest_d, "<"), d, nearest_d)
    return nearest_d


def clear_pass(origin, mate, opponents, r_eff, direction_ok=None):
    """True if a straight pass origin→mate clears every opponent cone.

    Includes low-stam defenders — they can still intercept a loose pass.
    Anti-tackle may ghost low-stam bodies; passing must not.

    `direction_ok` short-circuits the whole question. A pass is a kick, and a
    kick fires along MoveTo, so the carrier must first TURN to face the
    receiver — which parks the held ball on that heading. If a defender already
    owns that angle the ball is gone before the pass exists, so the lane counts
    as intercepted immediately and none of the cone geometry below can rescue
    it. `titanium.anti_tackle.aim_is_safe` supplies that test.
    """
    direction = unit_or_zero(mate - origin)
    # Skip defenders further from the pass target than the ball: they cannot
    # intercept a kicked ball that reaches the teammate first.  Done inside
    # the graph by widening the cone for trailing opponents.
    margin = r_eff + Float(0.1)
    d_to_mate = Distance(origin, mate)
    invariants = []
    for o in opponents:
        ha = opponent_half_angle(o, origin, r_eff)
        toward, cos_a, engulfed = opponent_invariants(o, origin, r_eff, ha)
        too_far_behind = CompareFloats(Distance(o, mate), d_to_mate + margin, ">")
        engulfed = And(engulfed, Not(too_far_behind))
        cos_a = ConditionalSetFloat(too_far_behind, Float(2.0), cos_a)
        invariants.append((toward, cos_a, engulfed))
    legal = is_legal_direction(direction, invariants)
    return legal if direction_ok is None else And(legal, direction_ok(direction))


def best_escape_pass(me, mates, opponents, r_eff, direction_ok=None):
    """Nearest teammate with a clear pass lane. Returns (can_pass, aim_dir, mate_pos).

    `direction_ok` is passed straight through to `clear_pass`, so a receiver
    sitting on an angle a defender owns is rejected outright rather than being
    picked and then lost on the turn.
    """
    best_mate = mates[0]
    best_dir = unit_or_zero(mates[0] - me)
    best_d = Float(1e9)
    any_ok = Bool(False)
    for mate in mates:
        ok = clear_pass(me, mate, opponents, r_eff, direction_ok)
        d = Distance(me, mate)
        better = And(ok, Or(Not(any_ok), CompareFloats(d, best_d, "<")))
        best_mate = ConditionalSetVector3(better, mate, best_mate)
        best_dir = ConditionalSetVector3(better, unit_or_zero(mate - me), best_dir)
        best_d = ConditionalSetFloat(better, d, best_d)
        any_ok = Or(any_ok, ok)
    return any_ok, best_dir, best_mate


def best_pass(origin, mates, opponents, r_eff, direction_ok=None):
    """Nearest teammate with a clear pass lane. Returns (can_pass, aim_dir, mate_pos)."""
    return best_escape_pass(origin, mates, opponents, r_eff, direction_ok)


def unopposed_walk_in(origin, left_post, right_post, defenders):
    """Can `origin` walk the ball into the goal mouth before any defender arrives?

    Target = closest point on the usable goal-line segment between the posts
    (same mouth geometry as post/scoring eval) — not the goal center.

    Carrier walks at CARRIER_SPEED; defenders race at SPRINT_SPEED. Samples
    along the path catch cut-across intercepts, not only the finish.

    Returns `(unopposed, walk_target)`.
    """
    goal_pt = closest_point_on_goal_mouth(origin, left_post, right_post)
    total_dist = Distance(origin, goal_pt)
    total_time = total_dist / Float(CARRIER_SPEED)
    unopposed = Bool(True)
    for frac in (0.25, 0.5, 0.75, 1.0):
        sample_pt = origin + (goal_pt - origin) * Float(frac)
        sample_time = total_time * Float(frac)
        fastest_def = Distance(defenders[0], sample_pt) / Float(SPRINT_SPEED)
        for d in defenders[1:]:
            dt = Distance(d, sample_pt) / Float(SPRINT_SPEED)
            fastest_def = ConditionalSetFloat(CompareFloats(dt, fastest_def, "<"), dt, fastest_def)
        beaten_here = CompareFloats(fastest_def, sample_time, "<=")
        unopposed = And(unopposed, Not(beaten_here))
    return unopposed, goal_pt
