"""Pure geometric primitives: tangents, forbidden cones, safe directions.

Nothing here knows about ball physics, player roles, or game state — every
function takes positions/radii and returns positions/directions. This is
the namespace every other module borrows its avoidance geometry from.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import ANGLE_EPS
from titanium.nodefn import graph_function


def pos(label: str):
    return RelativePosition(SoccerGetTransform(label), "Self")


def _rotate_xz_body(v, angle):
    """Rotation maths as plain inlined nodes.

    Separate from the `rotate_xz` graph function because a function body may not
    call another function (`titanium.nodefn` rule 3) — `_post_tangent_parts`
    needs the maths, not the call.
    """
    c = Cos(angle)
    s = Sin(angle)
    return Vector3(v.x * c - v.z * s, Float(0), v.x * s + v.z * c)


@graph_function("TiRotateXZ", ("Vector3", "Float"), "Vector3")
def rotate_xz(v, angle):
    """Rotate pitch vector around Y by `angle` radians (X/Z plane)."""
    return _rotate_xz_body(v, angle)


@cache
def unit_or_zero(v):
    return Normalize(v)


@cache
def clamp01(x):
    # Native ClampFloat — one node where this used to expand to four.
    return ClampFloat(x, Float(0), Float(1))


@cache
def opponent_invariants(opp_pos, ball, r_eff, half_angle):
    """Per-opponent values `direction_forbidden` needs that do NOT depend on
    the candidate direction being tested — and, crucially, not on `me`
    either. `nearest_safe_direction` is called separately per player (P1,
    P2, P3, GK clear, support lane, ...) with the same `ball`/`opponents`/
    `r_eff` every time; `@cache` here means only the FIRST of those calls
    per opponent actually builds these nodes, every later call across every
    other player just gets the same cached result back.
    """
    offset = opp_pos - ball
    dist = Magnitude(offset)
    toward = unit_or_zero(offset)
    # Inside circle => every direction is unsafe for this opponent.
    engulfed = CompareFloats(dist, r_eff, "<=")
    cos_a = Cos(half_angle)
    return toward, cos_a, engulfed


@graph_function("TiConeBlocks", ("Vector3", "Vector3", "Float", "Bool"), "Bool")
def _cone_blocks(direction, toward, cos_a, engulfed):
    """Body of `direction_forbidden` — the most-instantiated shape in the graph
    (every candidate direction x every opponent).

    The per-opponent `invariant` deliberately stays hoisted OUTSIDE: it is
    shared across every candidate tested from one origin, so pulling it in here
    would recompute it per call (`titanium.nodefn` rule 1).
    """
    inside = CompareFloats(DotProduct(direction, toward), cos_a, ">=")
    return Or(engulfed, inside)


def direction_forbidden(direction, invariant):
    """True if unit `direction` lies inside the opponent's forbidden cone.
    `invariant` is this opponent's precomputed `(toward, cos_a, engulfed)`
    from `opponent_invariants` — the only part that varies per candidate is
    the dot-product threshold check below."""
    toward, cos_a, engulfed = invariant
    return _cone_blocks(direction, toward, cos_a, engulfed)


@cache
def opponent_half_angle(opp_pos, ball, r_eff):
    dist = Magnitude(opp_pos - ball)
    ratio = clamp01(r_eff / ConditionalSetFloat(CompareFloats(dist, Float(1e-3), "<"), Float(1e-3), dist))
    return Asin(ratio) + Float(ANGLE_EPS)


@cache
def opponent_tangents(opp_pos, ball, half_angle):
    toward = unit_or_zero(opp_pos - ball)
    left = rotate_xz(toward, half_angle)
    right = rotate_xz(toward, Float(0) - half_angle)
    return unit_or_zero(left), unit_or_zero(right)


@cache
def post_tangent(origin, post_center, contact_r):
    """True boundary of a straight-line shot from `origin` around a post's
    physical body (radius `contact_r` = post + ball radius), on the side
    facing the goal's center line.

    A shot aimed at the raw post position (or the goal-mouth corner) would
    clip the post, not score — this is the actual max/min-angle edge. Same
    tangent-circle construction as `opponent_tangents` above (rotate the
    line-of-sight by +-asin(r/dist)), except here we want only the single
    tangent that faces the goal center, not both cone edges, so we pick
    whichever of the two candidates lands closer to z=0 by comparing the
    resulting points directly (no assumption about which post is "left").

    Returns `(direction, point)`: unit direction from `origin`, and the
    tangent point itself (handy for debug drawing / literal aim targets).
    """
    r = Float(contact_r)
    return post_tangent_dir(origin, post_center, r), post_tangent_point(origin, post_center, r)


def _post_tangent_parts(origin, post_center, contact_r):
    """Shared maths for the two post-tangent functions below.

    Split into a `_dir` and a `_point` function because a graph function returns
    exactly one value. That split is free in a shipped build: only `debug_viz`
    ever wants the tangent POINT, and release strips it, so
    `TiPostTangentPoint` is simply never emitted.
    """
    offset = post_center - origin
    dist = Magnitude(offset)
    safe_dist = ConditionalSetFloat(CompareFloats(dist, Float(1e-3), "<"), Float(1e-3), dist)
    toward = unit_or_zero(offset)
    alpha = Asin(clamp01(contact_r / safe_dist))
    cand_a = _rotate_xz_body(toward, alpha)
    cand_b = _rotate_xz_body(toward, Float(0) - alpha)
    l_sq = dist * dist - contact_r * contact_r
    l = Sqrt(ConditionalSetFloat(CompareFloats(l_sq, Float(0), "<"), Float(0), l_sq))
    point_a = origin + cand_a * l
    point_b = origin + cand_b * l
    inward_is_a = CompareFloats(Abs(point_a.z), Abs(point_b.z), "<=")
    return inward_is_a, cand_a, cand_b, point_a, point_b


@graph_function("TiPostTangentDir", ("Vector3", "Vector3", "Float"), "Vector3")
def post_tangent_dir(origin, post_center, contact_r):
    inward, cand_a, cand_b, _pa, _pb = _post_tangent_parts(origin, post_center, contact_r)
    return ConditionalSetVector3(inward, cand_a, cand_b)


@graph_function("TiPostTangentPoint", ("Vector3", "Vector3", "Float"), "Vector3")
def post_tangent_point(origin, post_center, contact_r):
    inward, _a, _b, point_a, point_b = _post_tangent_parts(origin, post_center, contact_r)
    return ConditionalSetVector3(inward, point_a, point_b)


def is_legal_direction(direction, invariants):
    bad = Bool(False)
    for invariant in invariants:
        bad = Or(bad, direction_forbidden(direction, invariant))
    return Not(bad)


def pick_better(cond_prefer_a, a, b):
    return ConditionalSetVector3(cond_prefer_a, a, b)


def nearest_safe_direction(ball, desired_target, opponents, r_eff):
    """Return unit direction closest (by dot) to desired among legal candidates."""
    desired = unit_or_zero(desired_target - ball)
    half_angles = [opponent_half_angle(o, ball, r_eff) for o in opponents]
    tangents = []
    invariants = []
    for opp, alpha in zip(opponents, half_angles):
        left, right = opponent_tangents(opp, ball, alpha)
        tangents.extend([left, right])
        invariants.append(opponent_invariants(opp, ball, r_eff, alpha))

    candidates = [desired] + tangents

    # Seed with desired if legal, else first legal tangent, else desired fallback.
    best = desired
    best_score = DotProduct(desired, desired)  # 1 if unit
    best_legal = is_legal_direction(desired, invariants)

    for cand in candidates[1:]:
        legal = is_legal_direction(cand, invariants)
        score = DotProduct(cand, desired)
        better = And(
            legal,
            Or(
                Not(best_legal),
                CompareFloats(score, best_score, ">"),
            ),
        )
        best = ConditionalSetVector3(better, cand, best)
        best_score = ConditionalSetFloat(better, score, best_score)
        best_legal = Or(best_legal, legal)

    return best


def clamp_own_half(p, team_goal):
    """Keep a point on our side of the halfway line.

    The keeper is allowed to press the carrier aggressively, but stepping
    over midfield to do it means an interception behind him is an open net.
    Team space is not sign-stable across home/away, so derive the side from
    our own goal's x rather than assuming one.
    """
    defends_negative = CompareFloats(team_goal.x, Float(0), "<")
    clamp_neg = ConditionalSetFloat(CompareFloats(p.x, Float(0), ">"), Float(0), p.x)
    clamp_pos = ConditionalSetFloat(CompareFloats(p.x, Float(0), "<"), Float(0), p.x)
    x = ConditionalSetFloat(defends_negative, clamp_neg, clamp_pos)
    return Vector3(x, p.y, p.z)


def closest_point_on_segment(origin, a, b):
    """Closest point on segment a→b to `origin` (pitch XZ)."""
    ab = b - a
    ab_len_sq = DotProduct(ab, ab)
    ab_len_sq = ConditionalSetFloat(CompareFloats(ab_len_sq, Float(1e-8), "<"), Float(1e-8), ab_len_sq)
    t = clamp01(DotProduct(origin - a, ab) / ab_len_sq)
    return a + ab * t


def closest_point_on_goal_mouth(origin, left_post, right_post):
    """Closest point on the usable goal line (segment between posts).

    Same mouth geometry scoring/eval uses: anything between the posts on the
    goal line is a walk-in finish; outside is wall. Walk toward this point —
    not the goal center — so the path is the shortest legal score.
    """
    return closest_point_on_segment(origin, left_post, right_post)
