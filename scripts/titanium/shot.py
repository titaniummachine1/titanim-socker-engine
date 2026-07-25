"""Shot legality and safe walking: is there a clean lane right now, and how
do we move without walking through a tackler."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import POST_AIM_RADIUS
from titanium.geometry import (
    is_legal_direction,
    nearest_safe_direction,
    opponent_half_angle,
    opponent_invariants,
    post_tangent,
    unit_or_zero,
)


def safe_walk_target(me, ball, desired, opponents, r_eff, step=5.5):
    direction = nearest_safe_direction(ball, desired, opponents, r_eff)
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
