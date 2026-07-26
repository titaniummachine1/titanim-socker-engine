"""Shot legality and walking: is there a clean lane right now, and how we
move toward a target (straight line — dodge logic lives in anti_tackle)."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import CARRIER_SPEED, POST_AIM_RADIUS, SPRINT_SPEED
from titanium.geometry import (
    closest_point_on_goal_mouth,
    is_legal_direction,
    opponent_half_angle,
    opponent_invariants,
    post_tangent,
    unit_or_zero,
)


def walk_target(me, desired, step=5.5):
    """Walk straight toward `desired` — no cone avoidance."""
    direction = unit_or_zero(desired - me)
    return me + direction * Float(step)


def clear_shot(shot_origin, opp_goal, opp_left_post, opp_right_post, opponents, r_eff):
    """Is there a legal straight-line lane into the enemy goal right now, and
    along which direction?

    Candidates are the three directions that actually score: straight at the
    goal centre, and the two post-tangent cone edges — the true min/max angle
    corrected for post+ball radius, since a ball aimed at the raw corner
    clips the post instead of going in. A candidate counts only if it also
    clears every opponent's forbidden cone, so the lane is genuinely open
    rather than merely goal-ward.

    ALL opponents count here (including exhausted ones): a released ball is
    loose and anyone can pick it up. Anti-tackle/dribble ghosts low-stam
    bodies; shooting must not.

    Centre is preferred whenever it is legal (largest margin for error); the
    tangents are the fallback for when somebody is standing in the middle.

    Returns `(lane_open, aim_direction)`.
    """
    dir_c = unit_or_zero(opp_goal - shot_origin)
    dir_l, _ = post_tangent(shot_origin, opp_left_post, POST_AIM_RADIUS)
    dir_r, _ = post_tangent(shot_origin, opp_right_post, POST_AIM_RADIUS)
    invariants = [
        opponent_invariants(o, shot_origin, r_eff, opponent_half_angle(o, shot_origin, r_eff))
        for o in opponents
    ]
    best = dir_c
    ok = is_legal_direction(dir_c, invariants)
    for cand in (dir_l, dir_r):
        legal = is_legal_direction(cand, invariants)
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


def clear_pass(origin, mate, opponents, r_eff):
    """True if a straight pass origin→mate clears every opponent cone.

    Includes low-stam defenders — they can still intercept a loose pass.
    Anti-tackle may ghost low-stam bodies; passing must not.
    """
    direction = unit_or_zero(mate - origin)
    invariants = [
        opponent_invariants(o, origin, r_eff, opponent_half_angle(o, origin, r_eff))
        for o in opponents
    ]
    return is_legal_direction(direction, invariants)


def best_escape_pass(me, mates, opponents, r_eff):
    """Nearest teammate with a clear pass lane. Returns (can_pass, aim_dir, mate_pos)."""
    best_mate = mates[0]
    best_dir = unit_or_zero(mates[0] - me)
    best_d = Float(1e9)
    any_ok = Bool(False)
    for mate in mates:
        ok = clear_pass(me, mate, opponents, r_eff)
        d = Distance(me, mate)
        better = And(ok, Or(Not(any_ok), CompareFloats(d, best_d, "<")))
        best_mate = ConditionalSetVector3(better, mate, best_mate)
        best_dir = ConditionalSetVector3(better, unit_or_zero(mate - me), best_dir)
        best_d = ConditionalSetFloat(better, d, best_d)
        any_ok = Or(any_ok, ok)
    return any_ok, best_dir, best_mate


def best_pass(origin, mates, opponents, r_eff):
    """Nearest teammate with a clear pass lane. Returns (can_pass, aim_dir, mate_pos)."""
    return best_escape_pass(origin, mates, opponents, r_eff)


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
