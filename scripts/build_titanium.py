"""Build real-game Titanium.txt via AIGamePyLibrary.

Carrier: current-tick forbidden-cone avoidance (tangent selection).
GK: safe cover region, forward aggression, rare sprint, gated challenge.
Simulator is validation only — this graph is the real Unity deliverable.

CANONICAL LOCATION. This file is the competition engine and lives in the
PRIVATE titanim-socker-engine repo. It must never be committed to
aicomp-soccer-sim, which is a PUBLIC repo (that tree ignores it explicitly).
Building writes the graph out to two places: the live Unity save folder, and
the public sim's gitignored `data/titanium/` so the drill harnesses can load
it via `--gk data/titanium/Titanium.txt` without the source ever going with
it.
"""
from __future__ import annotations

import math
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[0] / "worldcupteams" / "AIGamePyLibrary"))
sys.path.insert(0, str(ROOT.parents[0] / "worldcupteams"))

from AIGamePyLibrary import *  # noqa: E402
from AIGamePyLibrary.lib import data as graph_data  # noqa: E402
import ball_trajectory_graph as traj  # noqa: E402

SAVES = (
    Path.home()
    / "AppData"
    / "LocalLow"
    / "Unicorn One"
    / "AIComp"
    / "Saves"
    / "Soccer"
    / "Titanium.txt"
)
# Built graph is also dropped into the public sim repo purely so the drill /
# headless harnesses can load it. That path is gitignored there — the graph
# is a build artifact, the source stays here.
LOCAL_OUT = ROOT.parents[0] / "aicomp-soccer-sim" / "data" / "titanium" / "Titanium.txt"
# Every build lands here first. SAVES/LOCAL_OUT (the "live" submission copy
# and the sim's gated-champion copy) are only touched by --promote, after a
# candidate has actually won its gate — a build must never silently
# overwrite whatever's currently accepted as strongest.
CANDIDATE_OUT = ROOT / "out" / "Titanium_candidate.txt"
# Side-by-side playtest save (does NOT overwrite live Titanium.txt).
TEST_SAVES = (
    Path.home()
    / "AppData"
    / "LocalLow"
    / "Unicorn One"
    / "AIComp"
    / "Saves"
    / "Soccer"
    / "Titanium_test.txt"
)
TEST_LOCAL = ROOT.parents[0] / "aicomp-soccer-sim" / "data" / "titanium" / "Titanium_test.txt"
BACKUPS_DIR = ROOT.parents[0] / "aicomp-soccer-sim" / "data" / "titanium" / "backups"

BALL_RADIUS = 0.4064
# Upright post world radius. Not exposed by any SoccerGet — hardcoded from
# the same confirmed/measured Unity geometry the offline sim uses (posts at
# world x=+-40.2, world radius 0.3; see aicomp-soccer-sim/docs/SOCCER_GAME_MODEL.md).
POST_RADIUS = 0.3
POST_CONTACT_RADIUS = POST_RADIUS + BALL_RADIUS
ANGLE_EPS = math.radians(0.5)
WALK_SPEED = 7.0
SPRINT_SPEED = 8.0
# Opponent carrier speed while running with the ball. Confirmed measurement
# (AIA_UPSTREAM_QUIRKS.md #11: walk ~7 m/s flat, no stamina throttle) — same
# figure as WALK_SPEED, reused directly rather than guessing a separate
# constant. There is no "Opponent Velocity" getter, so heading comes from the
# held ball offset instead (see `carrier_heading`).
CARRIER_SPEED = WALK_SPEED

# Kick launch physics. Not exposed by any SoccerGet — hardcoded from the same
# TimePlot-calibrated constants the offline sim uses (aicomp-soccer-sim
# SimParams::fallback). Used only to ESTIMATE the charge that produced an
# already-observed ball speed (and from it, an estimated airborne hang time)
# — never to predict our own kicks, which we already control directly.
KICK_SPEED_BASE = 10.0 / 9.0
KICK_SPEED_PER_CHARGE = 290.0 / 9.0
KICK_LIFT_BASE = -0.323
KICK_LIFT_PER_CHARGE = 6.6667
GRAVITY = 9.81
# Warmup is already done while they hold+interact; remaining to full is
# (1-charge)*shot_charge_time_s once charge has started climbing.
SHOT_CHARGE_TIME_S = 0.38
# "Fully charged" for release purposes. The engine clamps charge at exactly
# 1.0 and holds it there, so this only needs to clear float noise.
FULL_CHARGE = 0.99
# Horizontal launch speed at charge=1.0 — used only to time "can an opponent
# sprint-intercept this full-power shot before it reaches the goal".
FULL_SHOT_SPEED = KICK_SPEED_BASE + KICK_SPEED_PER_CHARGE
# Held-ball visual/physical offset ahead of facing (AIA_UPSTREAM_QUIRKS #hold_offset_m).
HOLD_OFFSET_M = 1.67
HOLD_PROXY = 0.55
# Opponents inside this multiple of interact_r trigger anti-tackle / lackey logic.
ANTI_TACKLE_NEAR_MULT = 1.5
# Binary-search window around the desired facing (±radians).
ANTI_TACKLE_SEARCH_RAD = math.radians(120)
# Lackey stands this far behind the carrier — close enough for a guaranteed outlet.
LACKEY_STANDOFF_M = 3.5


def pos(label: str):
    return RelativePosition(SoccerGetTransform(label), "Self")


def plot_xz(prefix: str, color: str, vec) -> None:
    """Log a position/direction to TimePlot (F1) as two channels, X and Z —
    Y is always ~0 on this pitch so it's not worth the extra channel. Kept
    deliberately minimal (no Y, no derived values) so this doesn't add
    enough TimePlot load to visibly lag the real game.
    """
    comps = Vector3Split(vec)
    TimePlot(String(f"{prefix}.X"), color, String(""), comps.x)
    TimePlot(String(f"{prefix}.Z"), color, String(""), comps.z)


@cache
def rotate_xz(v, angle):
    """Rotate pitch vector around Y by `angle` radians (X/Z plane)."""
    c = Cos(angle)
    s = Sin(angle)
    return Vector3(v.x * c - v.z * s, Float(0), v.x * s + v.z * c)


@cache
def unit_or_zero(v):
    return Normalize(v)


@cache
def clamp01(x):
    return ConditionalSetFloat(
        CompareFloats(x, Float(0), "<"),
        Float(0),
        ConditionalSetFloat(CompareFloats(x, Float(1), ">"), Float(1), x),
    )


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


@cache
def direction_forbidden(direction, invariant):
    """True if unit `direction` lies inside the opponent's forbidden cone.
    `invariant` is this opponent's precomputed `(toward, cos_a, engulfed)`
    from `opponent_invariants` — the only part that varies per candidate is
    the dot-product threshold check below."""
    toward, cos_a, engulfed = invariant
    inside = CompareFloats(DotProduct(direction, toward), cos_a, ">=")
    return Or(engulfed, inside)


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
    offset = post_center - origin
    dist = Magnitude(offset)
    safe_dist = ConditionalSetFloat(CompareFloats(dist, Float(1e-3), "<"), Float(1e-3), dist)
    toward = unit_or_zero(offset)
    alpha = Asin(clamp01(contact_r / safe_dist))
    cand_a = rotate_xz(toward, alpha)
    cand_b = rotate_xz(toward, Float(0) - alpha)
    l_sq = dist * dist - Float(contact_r * contact_r)
    l = Sqrt(ConditionalSetFloat(CompareFloats(l_sq, Float(0), "<"), Float(0), l_sq))
    point_a = origin + cand_a * l
    point_b = origin + cand_b * l
    inward_is_a = CompareFloats(Abs(point_a.z), Abs(point_b.z), "<=")
    direction = ConditionalSetVector3(inward_is_a, cand_a, cand_b)
    point = ConditionalSetVector3(inward_is_a, point_a, point_b)
    return direction, point


_EVENT_LEG_CHAIN_BUILT = False


def _build_event_leg_chain_function():
    """`TitaniumEventLegChain` — up to three chained friction+wall-bounce
    legs (two possible bounces), aggregated down to "does any leg cross
    the goal line at `goal_x`, and if so when/where". Mirrors
    `ball_trajectory_graph._build_trajectory_solve`'s inline 3-leg chain
    (`first`/`second`/`third = _event_leg(...)` built inside ONE function
    body) rather than calling the `_call_event_leg` wrapper repeatedly:
    chaining raw `_event_leg` calls inside a single function body is safe
    (pure graph-node composition, no SetVariable/GetVariable round trip),
    but chaining `CustomFunction("TitaniumEventLeg", ...)` calls is NOT —
    its side-channel outputs share one variable slot that the second call
    clobbers before the first is read (this caused a 0-10 loss previously).

    A wall-bounce that redirects the ball toward a goal is a real threat
    even though the bounce itself doesn't score — chasing only the first
    leg (as this used to) misses exactly that case.
    """
    global _EVENT_LEG_CHAIN_BUILT
    if _EVENT_LEG_CHAIN_BUILT:
        return
    traj.build_trajectory_physics_functions()
    fn = traj._begin_function("TitaniumEventLegChain")
    start_parts = Vector3Split(fn.Param1)
    velocity_parts = Vector3Split(fn.Param2)
    start = Vector3(start_parts.x, start_parts.y, start_parts.z)
    velocity = Vector3(velocity_parts.x, Float(0), velocity_parts.z)
    live_y = fn.Param3
    goal_x = fn.Param4

    leg1 = traj._event_leg(start, velocity, live_y)
    leg2 = traj._event_leg(leg1["contact"], leg1["bounce_velocity"], live_y)
    leg3 = traj._event_leg(leg2["contact"], leg2["bounce_velocity"], live_y)

    def _at_goal(leg):
        x = Vector3Split(leg["goal_point"]).x
        return And(leg["has_goal"], CompareFloats(Abs(x - goal_x), Float(0.5), "<="))

    leg1_at_goal = _at_goal(leg1)
    leg2_at_goal = And(leg1["has_wall"], _at_goal(leg2))
    leg3_at_goal = And(leg1["has_wall"], And(leg2["has_wall"], _at_goal(leg3)))

    is_threat = Or(leg1_at_goal, Or(leg2_at_goal, leg3_at_goal))
    time2 = leg1["event_time"] + leg2["event_time"]
    time3 = time2 + leg3["event_time"]
    time = ConditionalSetFloat(
        leg1_at_goal,
        leg1["event_time"],
        ConditionalSetFloat(leg2_at_goal, time2, ConditionalSetFloat(leg3_at_goal, time3, Float(0))),
    )
    point = ConditionalSetVector3(
        leg1_at_goal,
        leg1["goal_point"],
        ConditionalSetVector3(leg2_at_goal, leg2["goal_point"], ConditionalSetVector3(leg3_at_goal, leg3["goal_point"], start)),
    )

    SetVariable("_TitaniumEventLegChainIsThreat", is_threat)
    SetVariable("_TitaniumEventLegChainTime", time)
    traj._finish(
        fn,
        point,
        (
            fn.Param1, fn.Param2, fn.Param3, fn.Param4,
            start, velocity, leg1_at_goal, leg2_at_goal, leg3_at_goal,
            is_threat, time, point,
        ),
    )
    _EVENT_LEG_CHAIN_BUILT = True


@cache
def _call_event_leg_chain(start, velocity, live_y, goal_x):
    """Call the shared TitaniumEventLegChain function (up to 2 wall
    bounces). Its two side-channel outputs (`is_threat`, `time`) must be
    read immediately after this call, before any other call to this
    function in the same tick — same single-shared-variable-slot
    constraint as `_call_event_leg`."""
    _build_event_leg_chain_function()
    point = CustomFunction("TitaniumEventLegChain", start, velocity, live_y, goal_x)
    return {
        "is_threat": GetVariable("_TitaniumEventLegChainIsThreat"),
        "time": GetVariable("_TitaniumEventLegChainTime"),
        "point": point,
    }


def _airborne_hang_time(speed):
    """Estimate the airborne hang time for an already-observed, undecelerated
    ball speed. There's no `SoccerGet` for ball height, so we can't directly
    see whether a moving ball is still in the air — but horizontal kick
    speed and vertical lift both come from the same charge value, so back
    out an estimated charge from the observed speed, then the lift that
    same charge would have produced.

    Deliberately conservative: treats the ball as freshly launched and
    still fully airborne. Better to over-estimate a real deep shot's reach
    than under-estimate it and never react — under-estimating is exactly
    what let a real striker score through us every time in isolated 1v1
    testing (ground friction was applied from t=0, so a lofted shot looked
    like it died ~15-20m short of where it actually landed).
    """
    charge_est = clamp01((speed - Float(KICK_SPEED_BASE)) / Float(KICK_SPEED_PER_CHARGE))
    lift_raw = Float(KICK_LIFT_BASE) + Float(KICK_LIFT_PER_CHARGE) * charge_est
    lift_est = ConditionalSetFloat(CompareFloats(lift_raw, Float(0), "<"), Float(0), lift_raw)
    return (Float(2.0) * lift_est) / Float(GRAVITY)


@cache
def _own_goal_threat(ball, ball_vel, own_goal_x, goal_half_width):
    """Does this free ball's predicted path cross OUR OWN goal line within
    the mouth, and if so when/where?

    Two phases, in order:
      1. Airborne glide (no friction) for an estimated hang time — a ball
         can cross the goal-line X while still in the air (scoring only
         ever checks XZ position, never height), so this is checked first
         in its own right, not just as a delay before phase 2.
      2. `ball_trajectory_graph`'s shared, already-calibrated closed-form
         ground physics (friction deceleration + wall bounce) via
         `_event_leg`, chained for up to two segments, starting from the
         projected landing point — not from `ball` directly, which is what
         silently assumed the ball was already grounded.

    Returns `(is_threat, time, point)`: bool, seconds until the crossing,
    and the crossing point. If the ball doesn't reach either goal within
    the airborne phase or two grounded segments, `is_threat` is false and
    the other two values are meaningless.
    """
    live_y = ball.y
    velocity = Vector3(ball_vel.x, Float(0), ball_vel.z)
    speed = Magnitude(velocity)
    hang_time = _airborne_hang_time(speed)

    # Phase 1: airborne crossing check.
    #
    # `own_goal_x`/`goal_half_width` are already live Nodes (e.g. team_goal.x,
    # Abs(left_post.z)) — NOT wrapped in Float(...). Float(value) does
    # AddNode("Float", str(value)); str() on a Node falls back to __repr__
    # (no __str__ defined), producing garbage like "Node(type=...)" as the
    # literal modifier. The Rust interpreter's parse_float silently defaults
    # any unparseable modifier to 0.0 — so Float(a_node) doesn't wrap the
    # node's live value, it silently becomes the constant 0.0. This was the
    # actual bug behind the whole "is_threat never fires right" symptom:
    # every "at our goal line" check was comparing against 0.0, not ~±39.5.
    dx = own_goal_x - ball.x
    vx_safe = ConditionalSetFloat(CompareFloats(Abs(velocity.x), Float(1e-4), "<"), Float(1e-4), velocity.x)
    t_cross_raw = dx / vx_safe
    t_cross = ConditionalSetFloat(CompareFloats(t_cross_raw, Float(0), "<"), Float(0), t_cross_raw)
    right_direction = CompareFloats(dx * velocity.x, Float(0), ">")
    within_hang = CompareFloats(t_cross, hang_time, "<=")
    z_at_cross = ball.z + velocity.z * t_cross
    in_mouth = CompareFloats(Abs(z_at_cross), goal_half_width, "<=")
    airborne_threat = And(And(right_direction, within_hang), in_mouth)
    airborne_point = Vector3(own_goal_x, ball.y, z_at_cross)

    # Phase 2: hand off to the grounded solver from the projected landing
    # point (ball glides `hang_time` seconds undecelerated first).
    landing_point = ball + velocity * hang_time

    # Up to three grounded legs (two wall bounces). A shot that bounces off
    # a wall and then crosses our goal line IS a real threat regardless of
    # whether the bounce itself scores — it redirects the ball toward our
    # goal, so we chase it rather than only the direct leg.
    chain = _call_event_leg_chain(landing_point, velocity, live_y, own_goal_x)
    ground_threat = chain["is_threat"]
    ground_time = hang_time + chain["time"]
    ground_point = chain["point"]

    is_threat = Or(airborne_threat, ground_threat)
    time = ConditionalSetFloat(airborne_threat, t_cross, ground_time)
    point = ConditionalSetVector3(airborne_threat, airborne_point, ground_point)
    return is_threat, time, point


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


def safe_walk_target(me, ball, desired, opponents, r_eff, step=5.5):
    direction = nearest_safe_direction(ball, desired, opponents, r_eff)
    return me + direction * Float(step)


def _release_catchable(shot_origin, opponents, interact_r):
    """True if any opponent can grab the ball the instant it is released."""
    catchable = Bool(False)
    for opp in opponents:
        near = CompareFloats(Distance(opp, shot_origin), interact_r, "<=")
        catchable = Or(catchable, near)
    return catchable


def _opponent_can_intercept_shot(opp, origin, aim, goal_dist, interact_r):
    """Can this opponent sprint-touch the ball on the origin→goal segment
    before a full-charge shot arrives?

    Closest point on the finite segment (not an infinite ray — defenders
    past the goal must not veto a shot that has already crossed the line).
    Opponents behind the release point are handled by `_release_catchable`,
    not here.
    """
    along = DotProduct(opp - origin, aim)
    along_clamped = ConditionalSetFloat(
        CompareFloats(along, Float(0), "<"),
        Float(0),
        ConditionalSetFloat(CompareFloats(along, goal_dist, ">"), goal_dist, along),
    )
    closest = origin + aim * along_clamped
    dist = Distance(opp, closest)
    speed = ConditionalSetFloat(
        CompareFloats(Float(FULL_SHOT_SPEED), Float(1e-3), "<"),
        Float(1e-3),
        Float(FULL_SHOT_SPEED),
    )
    t_ball = along_clamped / speed
    gap = dist - interact_r
    gap = ConditionalSetFloat(CompareFloats(gap, Float(0), "<"), Float(0), gap)
    t_opp = gap / Float(SPRINT_SPEED)
    ahead = CompareFloats(along_clamped, Float(1e-3), ">")
    return And(ahead, CompareFloats(t_opp, t_ball, "<="))


def _shot_aim_clear(origin, aim, opponents, interact_r, goal_dist):
    blocked = Bool(False)
    for opp in opponents:
        blocked = Or(
            blocked,
            _opponent_can_intercept_shot(opp, origin, aim, goal_dist, interact_r),
        )
    return Not(blocked)


def clear_shot(shot_origin, opp_goal, opp_left_post, opp_right_post, opponents, interact_r):
    """Free shot into the enemy goal?

    Only two gates: (1) nobody in interact radius at release, (2) nobody can
    sprint-intercept the full-charge ball before it reaches the goal.
    Prefer centre, then post tangents, then soft mids.

    Returns `(lane_open, aim_direction)`.
    """
    dir_c = unit_or_zero(opp_goal - shot_origin)
    dir_l, _ = post_tangent(shot_origin, opp_left_post, POST_CONTACT_RADIUS)
    dir_r, _ = post_tangent(shot_origin, opp_right_post, POST_CONTACT_RADIUS)
    dir_ml = unit_or_zero(dir_c + dir_l)
    dir_mr = unit_or_zero(dir_c + dir_r)
    goal_dist = Distance(shot_origin, opp_goal)
    release_ok = Not(_release_catchable(shot_origin, opponents, interact_r))

    best = dir_c
    ok = Bool(False)
    for cand in (dir_c, dir_l, dir_r, dir_ml, dir_mr):
        legal = And(
            release_ok,
            _shot_aim_clear(shot_origin, cand, opponents, interact_r, goal_dist),
        )
        take = And(legal, Not(ok))
        best = ConditionalSetVector3(take, cand, best)
        ok = Or(ok, legal)
    return ok, best


# Desperado: ball must clear at least this many metres before an intercept
# counts as "too early". Later intercepts are acceptable under pressure.
DESPERADO_EARLY_M = 8.0
# Must still point somewhat toward the enemy goal (never dump back at our net).
DESPERADO_MIN_FORWARD = 0.35


def _intercepted_too_early(origin, aim, opponents, interact_r, early_m=DESPERADO_EARLY_M):
    bad = Bool(False)
    for opp in opponents:
        bad = Or(
            bad,
            _opponent_can_intercept_shot(opp, origin, aim, Float(early_m), interact_r),
        )
    return bad


def desperado_forward_shot(
    shot_origin, opp_goal, opp_left_post, opp_right_post, teammates, opponents, interact_r
):
    """Most-forward aim toward the enemy goal that is not intercepted too early.

    Prefers scoring-cone directions, then slight swings around the goal axis.
    Never picks a backward dump at our own half unless a teammate pass is the
    only clear forward option (handled as a separate candidate toward a mate
    with Dot(pass, goal_axis) >= DESPERADO_MIN_FORWARD).

    Returns `(ok, aim)`.
    """
    fwd = unit_or_zero(opp_goal - shot_origin)
    dir_c = fwd
    dir_l, _ = post_tangent(shot_origin, opp_left_post, POST_CONTACT_RADIUS)
    dir_r, _ = post_tangent(shot_origin, opp_right_post, POST_CONTACT_RADIUS)
    dir_ml = unit_or_zero(dir_c + dir_l)
    dir_mr = unit_or_zero(dir_c + dir_r)
    # Bounded swings around the attack axis (±~20° / ±~40°) — still forward.
    swing_a = Float(0.35)
    swing_b = Float(0.70)
    dir_a = unit_or_zero(rotate_xz(fwd, swing_a))
    dir_b = unit_or_zero(rotate_xz(fwd, Float(0) - swing_a))
    dir_c2 = unit_or_zero(rotate_xz(fwd, swing_b))
    dir_d = unit_or_zero(rotate_xz(fwd, Float(0) - swing_b))

    release_ok = Not(_release_catchable(shot_origin, opponents, interact_r))

    candidates = [dir_c, dir_l, dir_r, dir_ml, dir_mr, dir_a, dir_b, dir_c2, dir_d]
    # Forward passes to teammates that clear early intercept — only if the
    # pass itself still advances toward their goal.
    for mate in teammates:
        candidates.append(unit_or_zero(mate - shot_origin))

    best = fwd
    best_score = Float(-2)
    ok = Bool(False)
    for cand in candidates:
        forward = DotProduct(cand, fwd)
        forward_ok = CompareFloats(forward, Float(DESPERADO_MIN_FORWARD), ">=")
        clear_early = Not(
            _intercepted_too_early(shot_origin, cand, opponents, interact_r)
        )
        legal = And(release_ok, And(forward_ok, clear_early))
        better = And(
            legal,
            Or(Not(ok), CompareFloats(forward, best_score, ">")),
        )
        best = ConditionalSetVector3(better, cand, best)
        best_score = ConditionalSetFloat(better, forward, best_score)
        ok = Or(ok, legal)
    return ok, best


def tackle_duty(slot: int):
    """Should THIS player be the one to press Interact on an opponent carrier?

    A tackle drains `min(tackler_stam, carrier_stam)` from BOTH players and is
    won by the tackler iff `tackler_stam >= carrier_stam`. Two consequences,
    neither of which we were using while everyone simply spammed Interact:

      * Sending everybody in wastes our strongest players. Among those who
        would actually WIN the duel, the *cheapest* winner should go; the
        others keep their stamina, and with it their own immunity to being
        tackled (a full-stamina player cannot be dispossessed at all).
      * When nobody can win outright, a doomed tackle is still worth making:
        it costs the carrier exactly as much as it costs us. Feed the weakest
        player in first and the carrier is ground down until somebody who was
        previously below the bar now clears it.

    So: if anyone in range can win, the lowest-stamina winner tackles;
    otherwise the lowest-stamina player in range tackles as a sacrifice.

    Ties break by slot through a tiny index epsilon on the sort key, so
    exactly one player is ever designated — without it, two equal-stamina
    players would both see themselves as "the cheapest" and both pile in,
    which is the waste this is meant to prevent.
    """
    carrier_stam = SoccerGetFloat("Ball Carrier Stamina")
    near = [SoccerGetBool(f"Is Ball Nearby Team Player {i}") for i in range(1, 5)]
    stam = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 5)]
    key = [stam[i] + Float((i + 1) * 0.0001) for i in range(4)]
    win = [And(near[i], CompareFloats(stam[i], carrier_stam, ">=")) for i in range(4)]

    any_win = Bool(False)
    for w in win:
        any_win = Or(any_win, w)
    # Prefer winners when one exists; otherwise everyone in range is a
    # candidate sacrifice.
    eligible = [ConditionalSetBool(any_win, win[i], near[i]) for i in range(4)]

    me = slot - 1
    duty = eligible[me]
    for k in range(4):
        if k == me:
            continue
        cheaper = And(eligible[k], CompareFloats(key[k], key[me], "<"))
        duty = And(duty, Not(cheaper))
    return duty


def player_interact(player: int, has_ball: Node, shoot_now: Node):
    """Interact policy: hold maximum charge permanently, release only to shoot.

    The engine charges while Interact is true, clamps at 1.0 with no decay
    and no max-hold timer, and fires the kick the frame Interact goes false —
    along that frame's MoveTo, not facing. Crucially, charging costs nothing:
    stamina only drains while *sprinting*. So a carrier should sit on a fully
    charged shot indefinitely and spend it the instant a lane opens, instead
    of dumping the ball at a fixed charge threshold the way this used to
    (which fired at ~0.7 charge in whatever direction MoveTo happened to
    point, and fired again on reaching 1.0).

    Claiming splits by ball state. A LOOSE ball is free — no duel, no cost,
    so anyone in range grabs it. A ball HELD by an opponent costs a stamina
    duel, so only the designated tackler presses (see `tackle_duty`); the
    rest keep their stamina instead of throwing it away on duels they were
    always going to lose.
    """
    nearby = SoccerGetBool(f"Is Ball Nearby Team Player {player}")
    loose = SoccerGetBool("Is Ball Loose")
    opp_has = SoccerGetBool("Opponent Has Ball")
    hold_charge = And(has_ball, Not(shoot_now))
    grab = And(nearby, loose)
    tackle = And(And(nearby, opp_has), tackle_duty(player))
    return Or(hold_charge, Or(grab, tackle))


def build_carrier_move(
    me,
    ball,
    opp_goal,
    opp_left_post,
    opp_right_post,
    teammates,
    opponents,
    r_eff,
    interact_r,
    has_ball,
    charge,
):
    """With ball: hold full charge; anti-tackle rotate; shoot/pass only when safe.

    Priority when releasing:
      1. Clear goal shot (no release-catch, no full-path intercept).
      2. Escape pass to a teammate who wins the intercept race (delays tackle).
      3. Desperado forward dump under pressure (not intercepted too early).
    Between releases: MoveTo uses anti-tackle binary search so the held ball
    stays outside every nearby opponent's interact bubble.
    """
    fwd = unit_or_zero(opp_goal - me)
    lane_ok, aim_clear = clear_shot(
        me, opp_goal, opp_left_post, opp_right_post, opponents, interact_r
    )
    pass_ok, aim_pass = best_escape_pass(me, teammates, opponents, interact_r, fwd)
    desp_ok, aim_desp = desperado_forward_shot(
        me,
        opp_goal,
        opp_left_post,
        opp_right_post,
        teammates,
        opponents,
        interact_r,
    )
    ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    near_r = interact_r * Float(ANTI_TACKLE_NEAR_MULT)
    danger = CompareFloats(nearest_opponent_dist(opponents, me), near_r, "<=")
    pressure = CompareFloats(
        nearest_opponent_dist(opponents, me),
        interact_r * Float(2.5),
        "<=",
    )

    shoot_clear = And(lane_ok, ready)
    # Escape pass when a tackler is in the 1.5× bubble and anti-tackle alone
    # may not be enough — only if a teammate truly wins the race to the ball.
    shoot_pass = And(And(Not(lane_ok), And(pass_ok, danger)), ready)
    shoot_desp = And(
        And(Not(lane_ok), And(Not(pass_ok), And(desp_ok, pressure))),
        ready,
    )
    shoot_now = And(has_ball, Or(shoot_clear, Or(shoot_pass, shoot_desp)))
    aim = ConditionalSetVector3(
        lane_ok,
        aim_clear,
        ConditionalSetVector3(And(Not(lane_ok), pass_ok), aim_pass, aim_desp),
    )

    # Anti-tackle: binary-search ±180° from the facing we actually want
    # (toward goal), until the held ball sits outside every opponent's
    # interact radius. If full clearance is impossible, prefer drain-only
    # contact over a winning steal.
    my_stam = SoccerGetFloat("Ball Carrier Stamina")
    walk_dir = anti_tackle_facing(me, fwd, opponents, interact_r, my_stam)
    walk = me + walk_dir * Float(5.5)
    shoot_to = me + aim * Float(10)
    move = ConditionalSetVector3(shoot_now, shoot_to, walk)
    chase = safe_walk_target(me, ball, ball, opponents, r_eff, step=8.0)
    move = ConditionalSetVector3(has_ball, move, chase)
    return move, shoot_now


def nearest_opponent_body(opponents, me):
    """Closest opponent BODY to `me`, plus that distance."""
    nearest = opponents[0]
    best_d = Distance(me, opponents[0])
    for opp in opponents[1:]:
        d = Distance(me, opp)
        nearer = CompareFloats(d, best_d, "<")
        nearest = ConditionalSetVector3(nearer, opp, nearest)
        best_d = ConditionalSetFloat(nearer, d, best_d)
    return nearest, best_d


def face_away_from_nearest(me, opponents):
    """Unit facing 180° from the nearest opponent (held-ball swings off them)."""
    nearest, best_d = nearest_opponent_body(opponents, me)
    return unit_or_zero(me - nearest), best_d


def facing_contact_flags(me, facing, opponents, interact_r, carrier_stam):
    """Whether this facing puts the held ball inside anyone's interact radius.

    Returns (clear_all, clear_steal):
      clear_all   — ball outside every opponent's interact radius
      clear_steal — ball outside interact of anyone who would WIN the duel
                    (opp_stam >= ours); a weaker opp may still drain stamina
    """
    facing_u = unit_or_zero(facing)
    hold = Float(HOLD_OFFSET_M)
    # Anyone farther than hold+interact cannot reach the held ball.
    reach = hold + interact_r
    ball_now = me + facing_u * hold
    ball_next = me + facing_u * (hold + Float(WALK_SPEED) * Float(0.08))
    opp_stams = [SoccerGetFloat(f"Opponent Player {i} Stamina") for i in range(1, 5)]
    clear_all = Bool(True)
    clear_steal = Bool(True)
    for opp, stam in zip(opponents, opp_stams):
        can_reach = CompareFloats(Distance(me, opp), reach, "<=")
        hit_now = CompareFloats(Distance(ball_now, opp), interact_r, "<=")
        hit_next = CompareFloats(Distance(ball_next, opp), interact_r, "<=")
        hit = And(can_reach, Or(hit_now, hit_next))
        can_steal = CompareFloats(stam, carrier_stam, ">=")
        clear_all = And(clear_all, Not(hit))
        clear_steal = And(clear_steal, Not(And(hit, can_steal)))
    return clear_all, clear_steal


def _anti_tackle_sample_degrees():
    """Unrolled binary-search angles in ±180° around desired facing.

    Dyadic midpoints of [0, 180] (five levels → 11.25°), both signs — a
    graph-safe stand-in for a real binary-search loop over yaw. Keeps the
    node count bounded while still converging on the smallest safe turn.
    """
    mags = [0.0]
    for level in range(0, 5):
        step = 180.0 / (2**level)
        for k in range(1, 2**level + 1):
            m = k * step
            if m <= 180.0 + 1e-9 and all(abs(m - x) > 1e-4 for x in mags):
                mags.append(float(m))
    out = []
    for m in mags:
        out.append(m)
        if m > 1e-6:
            out.append(-m)
    return tuple(out)


def anti_tackle_facing(me, desired, opponents, interact_r, carrier_stam):
    """Binary-search ±180° from `desired` until the held ball is untackleable.

    Goal: rotate as little as possible from the facing we want to hold, such
    that the ball (hold-offset along facing) is outside every opponent's
    interact radius. If that is impossible, fall back to clear-of-stealers
    (drain-only), then least-bad. Within a tier, smallest |yaw| wins.
    """
    desired_u = unit_or_zero(desired)
    best = desired_u
    best_tier = Float(-1)
    best_abs = Float(999)
    best_dot = Float(-2)
    for deg in _anti_tackle_sample_degrees():
        cand = unit_or_zero(rotate_xz(desired_u, Float(math.radians(deg))))
        clear_all, clear_steal = facing_contact_flags(
            me, cand, opponents, interact_r, carrier_stam
        )
        tier = ConditionalSetFloat(
            clear_all,
            Float(2),
            ConditionalSetFloat(clear_steal, Float(1), Float(0)),
        )
        abs_deg = Float(abs(deg))
        dot = DotProduct(cand, desired_u)
        better_tier = CompareFloats(tier, best_tier, ">")
        same_tier = And(
            Not(CompareFloats(tier, best_tier, "<")),
            Not(CompareFloats(tier, best_tier, ">")),
        )
        better_abs = And(same_tier, CompareFloats(abs_deg, best_abs, "<"))
        same_abs = And(
            same_tier,
            And(
                Not(CompareFloats(abs_deg, best_abs, "<")),
                Not(CompareFloats(abs_deg, best_abs, ">")),
            ),
        )
        better_dot = And(same_abs, CompareFloats(dot, best_dot, ">"))
        better = Or(better_tier, Or(better_abs, better_dot))
        best = ConditionalSetVector3(better, cand, best)
        best_tier = ConditionalSetFloat(better, tier, best_tier)
        best_abs = ConditionalSetFloat(better, abs_deg, best_abs)
        best_dot = ConditionalSetFloat(better, dot, best_dot)
    return best


def best_escape_pass(me, teammates, opponents, interact_r, attack_fwd):
    """Pass to a teammate who reaches the ball before any opponent.

    Used when a tackler is inside the 1.5× bubble and we need to delay the
    loss. Aim must not be catchable at release, and no opponent may
    intercept the segment me→mate before the ball arrives.
    """
    release_ok = Not(_release_catchable(me, opponents, interact_r))
    best = attack_fwd
    best_score = Float(-2)
    ok = Bool(False)
    for mate in teammates:
        dist = Distance(me, mate)
        not_self = CompareFloats(dist, Float(0.75), ">")
        aim = unit_or_zero(mate - me)
        # Teammate (or any teammate on the path) wins: no enemy intercept.
        clear = _shot_aim_clear(me, aim, opponents, interact_r, dist)
        forward = DotProduct(aim, attack_fwd)
        legal = And(
            release_ok,
            And(not_self, And(clear, CompareFloats(forward, Float(-0.15), ">="))),
        )
        better = And(legal, Or(Not(ok), CompareFloats(forward, best_score, ">")))
        best = ConditionalSetVector3(better, aim, best)
        best_score = ConditionalSetFloat(better, forward, best_score)
        ok = Or(ok, legal)
    return ok, best


def lackey_station(carrier, opp_goal):
    """Close outlet behind the ball — always one guaranteed safe-pass body."""
    fwd = unit_or_zero(opp_goal - carrier)
    return carrier - fwd * Float(LACKEY_STANDOFF_M)


def post_intercept_threat_times(opponents, team_goal, interact_r, our_carrier, carrier_stam):
    """Clocks for: enemy reaches our carrier, steals, then shoots / relays.

    Tackle rule: winner needs `tackler_stam >= carrier_stam`. Opponents with
    strictly less stamina cannot take the ball — their steal clock is pushed
    to infinity so we neither cover nor panic about a duel they cannot win.
    (Sacrifice drains are ignored here; those do not produce a turnover yet.)

    Always computed; callers pick when this dominates the live ETA.
    """
    times = []
    opp_stams = [SoccerGetFloat(f"Opponent Player {i} Stamina") for i in range(1, 5)]
    impossible = Float(999.0)
    for i, opp in enumerate(opponents):
        gap = Distance(opp, our_carrier) - interact_r
        gap = ConditionalSetFloat(CompareFloats(gap, Float(0), "<"), Float(0), gap)
        t_arrive = gap / Float(SPRINT_SPEED)
        t_shot = threat_eta_to_goal(opp, team_goal)
        t_after = t_shot
        for mate in opponents:
            t_relay = threat_eta_pass_then_shot(opp, mate, team_goal)
            t_after = ConditionalSetFloat(
                CompareFloats(t_relay, t_after, "<"), t_relay, t_after
            )
        can_steal = CompareFloats(opp_stams[i], carrier_stam, ">=")
        times.append(
            ConditionalSetFloat(can_steal, t_arrive + t_after, impossible)
        )
    return times


def loose_race_threat_times(opponents, ball, team_goal, interact_r):
    """Clocks for: enemy wins the loose ball, then shoots / pass-then-shoots."""
    times = []
    for opp in opponents:
        gap = Distance(opp, ball) - interact_r
        gap = ConditionalSetFloat(CompareFloats(gap, Float(0), "<"), Float(0), gap)
        t_arrive = gap / Float(SPRINT_SPEED)
        t_shot = threat_eta_to_goal(opp, team_goal)
        t_after = t_shot
        for mate in opponents:
            t_relay = threat_eta_pass_then_shot(opp, mate, team_goal)
            t_after = ConditionalSetFloat(
                CompareFloats(t_relay, t_after, "<"), t_relay, t_after
            )
        times.append(t_arrive + t_after)
    return times


def gk_cover_stand(shot_origin, team_goal, left_post, right_post, interact_r, charge=None):
    """Safe stand between `shot_origin` and goal along the post-tangent bisector.

    Static geometry: d_seal = R/sin(α) is the *deepest* point that still
    touches both extreme shot rays. Standing there is the conservative seal.

    Charge-aware (held ball): opponent still needs (1−c)·0.38s to reach full
    power. In that window we can walk `vg·t` toward the shooter, so we stand
    at d_seal − recover (clamped above the interact bubble). They may release
    anytime at the current charge — that only makes the ball slower, so the
    static cone at the forward stand still covers; the recover term is what
    keeps us able to get back onto the full-power seal before they finish.
    """
    to_goal = unit_or_zero(team_goal - shot_origin)
    to_l, _ = post_tangent(shot_origin, left_post, POST_CONTACT_RADIUS)
    to_r, _ = post_tangent(shot_origin, right_post, POST_CONTACT_RADIUS)
    bisector = unit_or_zero(to_l + to_r)
    cos_posts = clamp01(DotProduct(to_l, to_r))
    sin_half = Sqrt((Float(1) - cos_posts) * Float(0.5))
    sin_half = ConditionalSetFloat(CompareFloats(sin_half, Float(0.05), "<"), Float(0.05), sin_half)
    d_seal = interact_r / sin_half
    dist_goal = Distance(shot_origin, team_goal)
    d_deep = ConditionalSetFloat(
        CompareFloats(dist_goal - Float(2.0), Float(0.5), "<"),
        Float(0.5),
        dist_goal - Float(2.0),
    )
    # Deepest legal static seal (never past the goal-mouth cushion).
    d_passive = ConditionalSetFloat(CompareFloats(d_seal, d_deep, "<"), d_seal, d_deep)
    # Never stand inside the carrier's interact bubble (nutmeg / free turn).
    d_near = interact_r * Float(1.15)

    if charge is None:
        d_forward = d_passive
    else:
        c = clamp01(charge)
        t_to_full = (Float(1.0) - c) * Float(SHOT_CHARGE_TIME_S)
        recover = Float(WALK_SPEED) * t_to_full
        d_fwd = d_passive - recover
        d_forward = ConditionalSetFloat(CompareFloats(d_fwd, d_near, "<"), d_near, d_fwd)

    cover = shot_origin + bisector * d_forward
    cover_home = team_goal - to_goal * Float(2.0)

    def seals(stand):
        """Stand is inside the guaranteed intercept region (static miss test)."""
        to_g = unit_or_zero(stand - shot_origin)
        between = CompareFloats(Distance(stand, team_goal), Distance(shot_origin, team_goal), "<=")
        dist_sb = Magnitude(stand - shot_origin)
        cross_l = Abs(to_g.x * to_l.z - to_g.z * to_l.x)
        cross_r = Abs(to_g.x * to_r.z - to_g.z * to_r.x)
        cut_l = CompareFloats(cross_l * dist_sb, interact_r, "<=")
        cut_r = CompareFloats(cross_r * dist_sb, interact_r, "<=")
        return And(between, And(cut_l, cut_r))

    cover = ConditionalSetVector3(seals(cover), cover, cover_home)
    cover = ConditionalSetVector3(seals(cover), cover, cover_home)
    return cover, seals


def threat_cover(opp_pos, team_goal, left_post, right_post, interact_r):
    """Seal point for ONE opponent's shot cone at our goal.

    This is the whole "multi-body goalkeeper" idea: the exact construction
    the keeper uses against the ball carrier, handed to an outfield player
    and pointed at a different opponent. Every opponent who could receive and
    shoot gets somebody standing on the closest point that still covers both
    extremes of *their* scoring cone, so the team collectively seals every
    lane instead of one keeper guessing which one matters.
    """
    cover, _seals = gk_cover_stand(opp_pos, team_goal, left_post, right_post, interact_r)
    return cover


def clamp_own_half(p, team_goal):
    """Keep a point on our side of the halfway line (champion GK behaviour).

    Team Goal Center is WORLD (home ≈ −40, away ≈ +40); defended side follows
    its sign.
    """
    defends_negative = CompareFloats(team_goal.x, Float(0), "<")
    clamp_neg = ConditionalSetFloat(CompareFloats(p.x, Float(0), ">"), Float(0), p.x)
    clamp_pos = ConditionalSetFloat(CompareFloats(p.x, Float(0), "<"), Float(0), p.x)
    x = ConditionalSetFloat(defends_negative, clamp_neg, clamp_pos)
    return Vector3(x, p.y, p.z)


def flight_eta(origin, target):
    """Seconds for a full-charge ball to cover origin→target (straight)."""
    speed = ConditionalSetFloat(
        CompareFloats(Float(FULL_SHOT_SPEED), Float(1e-3), "<"),
        Float(1e-3),
        Float(FULL_SHOT_SPEED),
    )
    return Distance(origin, target) / speed


def threat_eta_to_goal(origin, goal):
    """ETA for a full-charge shot from `origin` to land in `goal`."""
    return flight_eta(origin, goal)


def threat_eta_pass_then_shot(passer, receiver, goal):
    """ETA for pass passer→receiver, then full-charge shot receiver→goal."""
    return flight_eta(passer, receiver) + flight_eta(receiver, goal)


def urgency_rank(i, times):
    """How many threats are more urgent than times[i] (0 = soonest).

    Ties break toward the lower index so ranks are a unique permutation.
    """
    rank = Float(0)
    ti = times[i]
    for j, tj in enumerate(times):
        if j == i:
            continue
        sooner = CompareFloats(tj, ti, "<")
        if j < i:
            equal = And(
                Not(CompareFloats(tj, ti, "<")),
                Not(CompareFloats(ti, tj, "<")),
            )
            sooner = Or(sooner, equal)
        rank = rank + ConditionalSetFloat(sooner, Float(1), Float(0))
    return rank


def threat_times_and_covers(
    opponents,
    has_bits,
    carrier,
    team_goal,
    left_post,
    right_post,
    interact_r,
    our_carrier,
    team_has,
    opp_has,
    ball,
    loose,
):
    """Per-opponent ETA + seal, for EVERY possession state.

    - Enemy has ball: 2-ply from their carrier (direct / pass-then-shot).
    - We have ball: steal ETA + threats generated after that intercept.
    - Loose: race-to-ball ETA + threats after they claim it.

    Always a live clock — not only when they already own the ball.
    """
    times = []
    covers = []
    # Stamina of OUR holder (by has-bit), not Ball Carrier Stamina — that
    # label tracks whoever currently holds, including the opponent.
    our_stams = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 5)]
    our_has = [SoccerGetBool(f"Team Player {i} Has Ball") for i in range(1, 5)]
    our_carrier_stam = our_stams[0]
    for i in range(4):
        our_carrier_stam = ConditionalSetFloat(our_has[i], our_stams[i], our_carrier_stam)
    post = post_intercept_threat_times(
        opponents, team_goal, interact_r, our_carrier, our_carrier_stam
    )
    race = loose_race_threat_times(opponents, ball, team_goal, interact_r)
    for i, opp in enumerate(opponents):
        t_direct = threat_eta_to_goal(opp, team_goal)
        t_relay = threat_eta_pass_then_shot(carrier, opp, team_goal)
        t_possess = ConditionalSetFloat(has_bits[i], t_direct, t_relay)
        t = ConditionalSetFloat(
            opp_has,
            t_possess,
            ConditionalSetFloat(team_has, post[i], race[i]),
        )
        t = ConditionalSetFloat(Or(opp_has, Or(team_has, loose)), t, t_possess)
        times.append(t)
        covers.append(threat_cover(opp, team_goal, left_post, right_post, interact_r))
        TimePlot(String(f"Titanium.Threat{i+1}.Eta"), "Red", String(""), t)
        TimePlot(String(f"Titanium.Threat{i+1}.StealEta"), "Orange", String(""), post[i])
        TimePlot(String(f"Titanium.Threat{i+1}.LooseEta"), "Yellow", String(""), race[i])
    return times, covers


def stand_seals_shot(stand, shot_origin, team_goal, left_post, right_post, interact_r):
    """True if `stand` already cuts both post-tangent extremes of this shot."""
    to_l, _ = post_tangent(shot_origin, left_post, POST_CONTACT_RADIUS)
    to_r, _ = post_tangent(shot_origin, right_post, POST_CONTACT_RADIUS)
    to_g = unit_or_zero(stand - shot_origin)
    between = CompareFloats(Distance(stand, team_goal), Distance(shot_origin, team_goal), "<=")
    dist_sb = Magnitude(stand - shot_origin)
    cross_l = Abs(to_g.x * to_l.z - to_g.z * to_l.x)
    cross_r = Abs(to_g.x * to_r.z - to_g.z * to_r.x)
    cut_l = CompareFloats(cross_l * dist_sb, interact_r, "<=")
    cut_r = CompareFloats(cross_r * dist_sb, interact_r, "<=")
    return And(between, And(cut_l, cut_r))


def threat_already_sealed(teammates, shot_origin, team_goal, left_post, right_post, interact_r):
    """Any teammate body already seals this opponent's shot cone."""
    sealed = Bool(False)
    for mate in teammates:
        sealed = Or(
            sealed,
            stand_seals_shot(mate, shot_origin, team_goal, left_post, right_post, interact_r),
        )
    return sealed


def unsealed_urgency_rank(i, times, sealed_flags):
    """Urgency rank among threats that are NOT already sealed (0 = soonest open).

    Sealed threats get a huge rank so nobody is assigned to babysit them.
    """
    huge = Float(99)
    base = ConditionalSetFloat(sealed_flags[i], huge, Float(0))
    # If sealed, skip counting — return huge.
    # If open: count other open threats that are more urgent.
    rank = Float(0)
    ti = times[i]
    for j, tj in enumerate(times):
        if j == i:
            continue
        # j beats i in urgency (same tie-break as urgency_rank)
        sooner = CompareFloats(tj, ti, "<")
        if j < i:
            equal = And(
                Not(CompareFloats(tj, ti, "<")),
                Not(CompareFloats(ti, tj, "<")),
            )
            sooner = Or(sooner, equal)
        open_j = Not(sealed_flags[j])
        rank = rank + ConditionalSetFloat(And(open_j, sooner), Float(1), Float(0))
    return ConditionalSetFloat(sealed_flags[i], huge, rank)


def cover_or_intercept(
    want_rank,
    times,
    covers,
    sealed_flags,
    me,
    ball,
    ball_vel,
):
    """Cover the want_rank-th still-open threat; else free to intercept the ball.

    Once every danger cone is already sealed by a teammate, spare defenders
    stop standing on redundant marks and go win the ball instead.
    """
    intercept = predict_ball_meet_point(me, ball, ball_vel, SPRINT_SPEED)
    move = intercept
    assigned = Bool(False)
    for i, cover in enumerate(covers):
        mine = CompareFloats(
            unsealed_urgency_rank(i, times, sealed_flags), Float(want_rank), "=="
        )
        take = And(mine, Not(assigned))
        move = ConditionalSetVector3(take, cover, move)
        assigned = Or(assigned, mine)
    # If nothing open left for this slot, stay on intercept.
    move = ConditionalSetVector3(assigned, move, intercept)
    return move, assigned


def aggressive_press_with_cover(me, cover_move, on_cover, carrier, interact_r):
    """Presser: always press the holder; seal only if cheap.

    Against a stalling midfield carrier the low-urgency seal is often a
    camping spot far from the ball. Primary job is to go win it. If the
    assigned seal is nearly on the way to the carrier, or already sits in
    their interact bubble / shot apex, take that point first — press and
    a leftover cone at once.
    """
    direct = Distance(me, carrier)
    via = Distance(me, cover_move) + Distance(cover_move, carrier)
    detour = via - direct
    cheap_detour = CompareFloats(detour, Float(5.0), "<=")
    seal_near_carrier = CompareFloats(
        Distance(cover_move, carrier), interact_r * Float(2.5), "<="
    )
    use_seal = And(on_cover, Or(cheap_detour, seal_near_carrier))
    return ConditionalSetVector3(use_seal, cover_move, carrier)


def outfield_press_roles(outfield, carrier_stam, carrier_body):
    """Who presses vs who covers, with a clean role swap.

    Nominal flanker is P3 (least-urgent cone). Only an outfielder who can
    *win* the stamina duel (`stam >= carrier_stam`) may be sent to tackle.
    Prefer P3 when they qualify; otherwise the closest capable outfielder
    presses, and P3 inherits that player's cover urgency so the marks stay
    filled. If nobody can win the duel, nobody presses — never send a body
    that will only burn stamina and lose.
    """
    # slots 1..3 only (GK keeps its own press path)
    stam = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 4)]
    can = [CompareFloats(stam[i], carrier_stam, ">=") for i in range(3)]
    dist = [Distance(outfield[i], carrier_body) for i in range(3)]
    any_can = Or(Or(can[0], can[1]), can[2])

    # Prefer capable P3; else closest capable among P1/P2.
    # Distance key + tiny slot epsilon → exactly one presser.
    key = [dist[i] + Float((i + 1) * 0.0001) for i in range(3)]
    prefer3 = can[2]
    is_p3 = And(any_can, prefer3)
    p1_beats_p2 = Or(Not(can[1]), CompareFloats(key[0], key[1], "<"))
    is_p1 = And(any_can, And(Not(prefer3), And(can[0], p1_beats_p2)))
    is_p2 = And(any_can, And(Not(prefer3), And(can[1], Not(is_p1))))
    is_presser = [is_p1, is_p2, is_p3]

    # Cover urgency remap: default P1→0, P2→1, P3→2.
    # When P1 presses, P3 takes urgency 0. When P2 presses, P3 takes 1.
    u1 = Float(0)
    u2 = Float(1)
    u3 = Float(2)
    u3 = ConditionalSetFloat(is_p1, Float(0), u3)
    u3 = ConditionalSetFloat(is_p2, Float(1), u3)
    return is_presser, [u1, u2, u3], any_can


def cover_for_urgency_rank(want_rank, times, covers):
    """MoveTo = seal for the threat whose urgency rank == want_rank (0=soonest)."""
    move = covers[0]
    for i, cover in enumerate(covers):
        mine = CompareFloats(urgency_rank(i, times), Float(want_rank), "==")
        move = ConditionalSetVector3(mine, cover, move)
    return move


def opponent_carrier(opponents, ball):
    """Opponent BODY position currently holding the ball, falling back to
    the nearest opponent to the ball when no explicit holder bit is set.
    Press/seal targets must always be a body position, never the ball's
    held-offset point."""
    has_bits = [SoccerGetBool(f"Opponent Player {i} Has Ball") for i in range(1, 5)]
    carrier = opponents[-1]
    has_any = Bool(False)
    for opp, has in zip(opponents, has_bits):
        carrier = ConditionalSetVector3(has, opp, carrier)
        has_any = Or(has_any, has)
    nearest = opponents[0]
    nearest_d = Distance(opponents[0], ball)
    for opp in opponents[1:]:
        d = Distance(opp, ball)
        nearer = CompareFloats(d, nearest_d, "<")
        nearest = ConditionalSetVector3(nearer, opp, nearest)
        nearest_d = ConditionalSetFloat(nearer, d, nearest_d)
    return ConditionalSetVector3(has_any, carrier, nearest)


def team_carrier(players):
    """Our BODY position currently holding the ball (no loose fallback)."""
    has_bits = [SoccerGetBool(f"Team Player {i} Has Ball") for i in range(1, 5)]
    carrier = players[-1]
    for p, has in zip(players, has_bits):
        carrier = ConditionalSetVector3(has, p, carrier)
    return carrier, has_bits


def draw_2ply_threat_tree(
    carrier,
    bodies,
    has_bits,
    goal_center,
    left_post,
    right_post,
    active,
    shot_color: str,
    pass_color: str,
    eta_prefix: str,
):
    """Debug-draw the immediate 2-ply possession tree from `carrier`.

    Always drawn (both sides) — ETA channels / cover logic decide which
    threats are live. Collapsing on possession made enemy trees vanish the
    moment we held the ball, which hid steal→shot threats.
    """
    # Force visible every tick; `active` kept for API compat but ignored.
    _ = active
    on = Bool(True)
    collapse = carrier

    def gate(p, _on=None):
        return p

    origin_on = carrier
    _, aim_l = post_tangent(carrier, left_post, POST_CONTACT_RADIUS)
    _, aim_r = post_tangent(carrier, right_post, POST_CONTACT_RADIUS)
    DebugDrawLine(origin_on, aim_l, Float(0.06), shot_color)
    DebugDrawLine(origin_on, aim_r, Float(0.06), shot_color)
    DebugDrawDisc(aim_l, Float(0.15), Float(0.05), shot_color)
    DebugDrawDisc(aim_r, Float(0.15), Float(0.05), shot_color)
    t_direct = threat_eta_to_goal(carrier, goal_center)
    TimePlot(String(f"{eta_prefix}.DirectShot.Eta"), shot_color, String(""), t_direct)

    for i, mate in enumerate(bodies):
        mate_on = Not(has_bits[i])
        # Still skip drawing self-pass to the carrier body.
        mate_pt = ConditionalSetVector3(mate_on, mate, carrier)
        DebugDrawLine(origin_on, mate_pt, Float(0.045), pass_color)
        DebugDrawDisc(mate_pt, Float(0.18), Float(0.04), pass_color)
        t_pass = ConditionalSetFloat(mate_on, flight_eta(carrier, mate), Float(0))
        TimePlot(String(f"{eta_prefix}.Pass{i+1}.Eta"), pass_color, String(""), t_pass)

        _, m_l = post_tangent(mate, left_post, POST_CONTACT_RADIUS)
        _, m_r = post_tangent(mate, right_post, POST_CONTACT_RADIUS)
        DebugDrawLine(mate_pt, ConditionalSetVector3(mate_on, m_l, carrier), Float(0.05), shot_color)
        DebugDrawLine(mate_pt, ConditionalSetVector3(mate_on, m_r, carrier), Float(0.05), shot_color)
        t_then_shot = ConditionalSetFloat(
            mate_on, threat_eta_pass_then_shot(carrier, mate, goal_center), Float(0)
        )
        TimePlot(
            String(f"{eta_prefix}.Pass{i+1}ThenShot.Eta"), shot_color, String(""), t_then_shot
        )

        for j, other in enumerate(bodies):
            if j == i:
                continue
            other_on = And(mate_on, Not(has_bits[j]))
            DebugDrawLine(
                mate_pt,
                ConditionalSetVector3(other_on, other, carrier),
                Float(0.03),
                pass_color,
            )
            t_relay = ConditionalSetFloat(
                other_on, flight_eta(carrier, mate) + flight_eta(mate, other), Float(0)
            )
            TimePlot(
                String(f"{eta_prefix}.Pass{i+1}to{j+1}.Eta"),
                pass_color,
                String(""),
                t_relay,
            )


@cache
def carrier_heading(carrier, ball):
    """Unit heading of the opponent carrying the ball.

    There is no "Opponent Player N Velocity" getter, but we don't need one:
    the engine parks a held ball at `body + facing * hold_offset`, and a
    player's `facing` tracks their MoveTo target. So the held-ball offset
    direction IS the carrier's heading, available from current-tick data with
    no cross-tick memory.

    (This is the same offset that made raw "Ball Velocity" useless while the
    ball is held — a spinning carrier swings the offset around. As a
    *direction* it is exactly what we want; as a velocity it was noise.)
    """
    return unit_or_zero(ball - carrier)


@cache
def carrier_lead(gk, carrier, ball, cover_now, charge):
    """Where the carrier will be when their shot can actually hurt us.

    Sealing the cone from where a carrier stands *right now* is what AIA
    exploits: it runs a straight line across the mouth, the cone apex slides
    sideways every tick, and the GK spends the whole run trailing its own
    target by its acceleration lag — never arriving, never tackling.

    The charge clock bounds how far it is safe to lead. A carrier at charge
    `c` physically cannot release a *full-power* shot for another
    `(1-c) * SHOT_CHARGE_TIME_S` seconds. They can of course release early,
    but an early release is a slower ball, which we have correspondingly more
    time to reach. So lead by whichever is smaller:

      * `t_reach` — how long we need to get onto the current cover point.
        Leading further than this is pointless; we cannot use the extra time.
      * `t_full`  — when a full-power shot first becomes possible. Leading
        past this would abandon a cone that is about to go live.

    The clamp makes the behaviour collapse continuously: a carrier just
    running (`c ~ 0`) is led by the full ~0.38s (~2.9m, enough to actually
    meet them), while a carrier winding up (`c -> 1`) is led by ~0 and we
    simply cover where they are. No discrete "press vs seal" mode switch —
    those discontinuous jumps were what AIA exploited before.
    """
    heading = carrier_heading(carrier, ball)
    t_reach = Distance(gk, cover_now) / Float(WALK_SPEED)
    t_full = (Float(1.0) - clamp01(charge)) * Float(SHOT_CHARGE_TIME_S)
    lead_t = ConditionalSetFloat(CompareFloats(t_reach, t_full, "<"), t_reach, t_full)
    return carrier + heading * (Float(CARRIER_SPEED) * lead_t)


@cache
def predict_ball_meet_point(me, ball, ball_vel, speed):
    """Where to walk to meet a moving ball, instead of its live position.

    Closed-form interception solve. We want the earliest t >= 0 where our own
    reach equals the ball's distance:

        |ball + v*t - me| = speed * t

    Squaring both sides eliminates the direction entirely and leaves a plain
    quadratic in t (d = ball - me):

        (|v|^2 - speed^2) t^2 + 2 (d.v) t + |d|^2 = 0

    which is solved exactly, once, with no iteration.

    This REPLACES a 3-step fixed-point iteration (t <- |ball + v*t - me| /
    speed) that looked equivalent but is only a contraction while
    |v| < speed. Above that it diverges geometrically at ratio |v|/speed --
    and a real max-power kick is ~29 m/s against a 7 m/s walk, so three
    steps blew the target up by (29/7)^3 ~ 71x. Confirmed in a real-game
    TimePlot capture (2026-07-25): meet points up to 1085 m out on an 80x50
    pitch, on 1.5-9.4% of ticks, dragging players (and their stamina, via
    the contested-sprint branch) toward nothing.

    Two ways there is no straight-line answer, both ending at the same
    sane fallback -- where the ball is actually going to stop:

      * discriminant < 0, or both roots negative: the ball is simply
        outrunning us on this heading and no meeting point exists.
      * the solve returns a time later than the ball's own stop time
        (t_stop = |v| / friction): the maths is answering with a point the
        ball never reaches, because it has come to rest first.

    Still a friction-free straight line for the intercept itself (matching
    what this function has always assumed), so it slightly overshoots on a
    long-range shot; the stop-point clamp is what keeps that bounded to
    somewhere the ball can physically be. Wall bounces are not modelled --
    a bounce only ever brings the ball back toward us, never further away.
    """
    v = Vector3(ball_vel.x, Float(0), ball_vel.z)
    d = ball - me
    s = Float(speed)

    vv = DotProduct(v, v)
    dv = DotProduct(d, v)
    dd = DotProduct(d, d)

    a = vv - s * s
    b = Float(2.0) * dv
    c = dd

    disc = b * b - Float(4.0) * a * c
    has_real = CompareFloats(disc, Float(0), ">=")
    # Clamp before the root so a negative discriminant can never produce NaN;
    # `has_real` is what actually gates the result.
    sqrt_disc = Sqrt(ConditionalSetFloat(has_real, disc, Float(0)))

    # |v| == speed collapses the quadratic to a linear equation (b t + c = 0).
    degenerate = CompareFloats(Abs(a), Float(1e-4), "<")
    b_safe = ConditionalSetFloat(CompareFloats(Abs(b), Float(1e-4), "<"), Float(1e-4), b)
    t_linear = (Float(0) - c) / b_safe
    linear_ok = CompareFloats(t_linear, Float(0), ">=")

    two_a = ConditionalSetFloat(degenerate, Float(1e-4), Float(2.0) * a)
    r1 = ((Float(0) - b) - sqrt_disc) / two_a
    r2 = ((Float(0) - b) + sqrt_disc) / two_a
    first = CompareFloats(r1, r2, "<=")
    lo = ConditionalSetFloat(first, r1, r2)
    hi = ConditionalSetFloat(first, r2, r1)
    lo_ok = CompareFloats(lo, Float(0), ">=")
    hi_ok = CompareFloats(hi, Float(0), ">=")
    # Earliest non-negative root; `hi` only matters when `lo` is in the past.
    t_quad = ConditionalSetFloat(lo_ok, lo, hi)
    quad_ok = And(has_real, Or(lo_ok, hi_ok))

    t = ConditionalSetFloat(degenerate, t_linear, t_quad)
    solved = ConditionalSetBool(degenerate, linear_ok, quad_ok)

    # The solved t assumes a constant-velocity ball, so evaluate the POSITION
    # with the real decelerating displacement instead:
    #
    #     D(t) = |v| t - (1/2) f t^2      for 0 <= t <= t_stop = |v| / f
    #
    # D peaks exactly at t_stop (D' = |v| - f t) with D(t_stop) = |v|^2 / 2f,
    # the true friction stop distance. So clamping t into [0, t_stop] makes
    # one expression cover every case: a real intercept lands friction-
    # corrected on the path, and "no intercept exists" / "the answer is later
    # than the ball survives" both collapse to exactly the stop point. That
    # also bounds the output to at most one stop distance from the ball --
    # ~76 m at the 30 m/s speed cap, versus the 1085 m this used to emit.
    ball_speed = Magnitude(v)
    friction = Float(traj.FRICTION_ACCEL)
    t_stop = ball_speed / friction
    t_want = ConditionalSetFloat(solved, t, t_stop)
    t_eff = ConditionalSetFloat(CompareFloats(t_want, Float(0), "<"), Float(0), t_want)
    t_eff = ConditionalSetFloat(CompareFloats(t_eff, t_stop, ">"), t_stop, t_eff)

    return ball + v * t_eff


def nearest_opponent_dist(opponents, ball):
    nearest_d = Distance(opponents[0], ball)
    for opp in opponents[1:]:
        d = Distance(opp, ball)
        nearest_d = ConditionalSetFloat(CompareFloats(d, nearest_d, "<"), d, nearest_d)
    return nearest_d


def gk_policy(
    gk,
    ball,
    ball_vel,
    team_goal,
    left_post,
    right_post,
    opponents,
    r_eff,
    has_ball,
    charge,
    opp_has,
    loose,
    carrier_charge,
):
    """GK stays inside the shot-intercept safe region.

    Sprint only for a loose goal-bound ball that walk cannot reach in time.
    Held-ball cover uses the carrier body + their current shot charge so we
    step forward while they still have charge time left. Interact follows
    AIA Player Interact so any nearby claimable ball is always grabbed.
    """
    interact_r = SoccerGetFloat("Player Interact Radius")
    cover, seals = gk_cover_stand(ball, team_goal, left_post, right_post, interact_r)

    # With ball: clear upfield; do not sprint for the clear walk-up.
    #
    # Routed through the same forbidden-cone avoidance the outfielders use
    # (`safe_walk_target`) rather than a raw walk toward `clear_dir` — a GK
    # who just won the ball deep in his own box is standing right next to
    # whoever he took it from, and a blind walk toward the clear direction
    # can carry him straight past that opponent, handing back an instant
    # re-tackle a few metres from our own goal (confirmed via
    # titanium_matchtrace: exactly this sequence conceded a goal at t=36.8s
    # of an AIA3 match — GK wins the ball at x=-38, walks to clear, gets
    # retackled by the stationary opponent one tick later, ball walked in).
    clear_dir = SoccerGetVector3("Clear direction from Teammate 4")
    clear_target = gk + unit_or_zero(clear_dir) * Float(14)
    clear_target = ConditionalSetVector3(
        IsNull(clear_dir), pos("Opponent Goal Center"), clear_target
    )
    clear_spot = safe_walk_target(gk, ball, clear_target, opponents, r_eff, step=14)
    # Post-tackle: same 180° face-away as outfield — do not stroll through the
    # player you just dispossessed with the ball still on their side.
    away_gk, d_near_gk = face_away_from_nearest(gk, opponents)
    post_tackle_gk = CompareFloats(d_near_gk, interact_r * Float(1.4), "<=")
    escape_gk = gk + away_gk * Float(10)
    clear_spot = ConditionalSetVector3(post_tackle_gk, escape_gk, clear_spot)
    move = ConditionalSetVector3(has_ball, clear_spot, cover)

    # Sprint ONLY to intercept a free ball whose predicted path crosses our
    # own goal line — and only when walking can't already get there in time.
    # Gated on `loose`: held-ball "Ball Velocity" is the offset point, not
    # the carrier, and spinning opponents were fabricating fake threats.
    goal_half_width_est = Abs(left_post.z)
    is_threat_raw, threat_time, threat_point = _own_goal_threat(
        ball, ball_vel, team_goal.x, goal_half_width_est
    )
    is_threat = And(is_threat_raw, loose)
    dist_gk_threat = Distance(gk, threat_point)
    walk_time_threat = dist_gk_threat / Float(WALK_SPEED)
    sprint_time_threat = dist_gk_threat / Float(SPRINT_SPEED)
    walk_cant_make_it = CompareFloats(walk_time_threat, threat_time, ">")
    sprint_can_make_it = CompareFloats(sprint_time_threat, threat_time, "<=")
    sprint = And(is_threat, And(walk_cant_make_it, sprint_can_make_it))

    # A genuine goal-bound threat overrides static cover/clear — meet the
    # ball where the math says it will be, not where it is right now.
    move = ConditionalSetVector3(is_threat, threat_point, move)

    # Opponent holds the ball: charge-aware press from the carrier BODY
    # (never the held-ball offset — spin would yank the target around),
    # aimed at where they will BE, not where they are. See `carrier_lead`.
    carrier = opponent_carrier(opponents, ball)
    press_now, _seals_now = gk_cover_stand(
        carrier, team_goal, left_post, right_post, interact_r, charge=carrier_charge
    )
    carrier_pred = carrier_lead(gk, carrier, ball, press_now, carrier_charge)
    press, _carrier_seals = gk_cover_stand(
        carrier_pred, team_goal, left_post, right_post, interact_r, charge=carrier_charge
    )
    move = ConditionalSetVector3(opp_has, press, move)

    # When the ball is claimable (nearby / loose toward us), step onto it so
    # Interact can fire — do not stay glued to a cover point and let it pass.
    #
    # A genuine chase (not already in interact range) targets the ball's
    # predicted path, not its live position — chasing live position is a
    # pure-pursuit curve that bleeds ground to any ball with real velocity.
    # Once it's already `nearby` there's nothing left to lead, so that case
    # keeps the raw ball position.
    nearby = SoccerGetBool("Is Ball Nearby Team Player 4")
    closest_gk = SoccerGetBool("Is Team Player 4 Closest Teammate to Ball")
    meet_point_gk = predict_ball_meet_point(gk, ball, ball_vel, WALK_SPEED)
    chase_target_gk = ConditionalSetVector3(nearby, ball, meet_point_gk)
    go_to_ball = Or(And(loose, closest_gk), And(nearby, Not(has_ball)))
    chase_ok = Or(nearby, seals(gk))
    move = ConditionalSetVector3(And(go_to_ball, chase_ok), chase_target_gk, move)
    move = ConditionalSetVector3(And(nearby, Not(has_ball)), ball, move)

    # Champion behaviour: keep P4 on our half when not carrying (kick aim must
    # not be clamped — Interact release fires along MoveTo).
    move = ConditionalSetVector3(has_ball, move, clamp_own_half(move, team_goal))

    # Same permanent-full-charge policy as everyone else: hold the ball on a
    # maxed shot and only let go once the clear lane is actually open, rather
    # than punting it into the nearest opponent at a fixed charge threshold.
    clear_aim = unit_or_zero(
        ConditionalSetVector3(IsNull(clear_dir), pos("Opponent Goal Center") - gk, clear_dir)
    )
    # Same shot-origin correction as build_carrier_move: the kick fires from
    # the GK's own center, not the ball's visually-offset held position.
    clear_invariants = [
        opponent_invariants(o, gk, r_eff, opponent_half_angle(o, gk, r_eff))
        for o in opponents
    ]
    gk_ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    gk_shoot = And(has_ball, And(is_legal_direction(clear_aim, clear_invariants), gk_ready))
    interact = player_interact(4, has_ball, gk_shoot)
    return move, sprint, interact


def main() -> None:
    # Faceoff (team space: +X attack). Team space is already mirrored to
    # whichever physical side we're on, so two legality rules apply
    # regardless of home/away:
    #   1. No coordinate may be positive — that's past the halfway line
    #      into the opponent's half before kickoff even starts.
    #   2. Only the kickoff-taker may sit inside the center circle
    #      (r=7.25, confirmed in aicomp-soccer-sim/docs/AIA_UPSTREAM_QUIRKS.md
    #      #13). Every other body must be far enough out that its
    #      distance from center clears that radius.
    #
    # Magnitudes here are read directly out of AIA3.txt's own graph (traced
    # ConstructSoccerProperties's 4 inputs back through its
    # StrikerKickoffPos/DefenderKickoffPos/PlaymakerKickoffPos/
    # GoalieKickoffPos SetVariable chain to their literal Float values:
    # striker ~(1, 7), defender (5, 7), playmaker (11, 0), goalie (36, 0) —
    # mirrored negative here to stay in our own half.
    # P1 (kickoff-taker): distance 2.0 from center — inside, the one slot
    # allowed to be. P2 distance sqrt(5^2+7^2)=8.6, P3 distance 11, P4
    # distance 36 — all clear of the r=7.25 circle.
    InitializeSoccer(
        "Titanium_test",
        "Poland",
        Vector3(Float(0), Float(0), Float(2)),
        Vector3(Float(-5), Float(0), Float(7)),
        Vector3(Float(-11), Float(0), Float(0)),
        Vector3(Float(-36), Float(0), Float(0)),
    )

    ball = pos("Ball")
    opp_goal = pos("Opponent Goal Center")
    team_goal = pos("Team Goal Center")
    left_post = pos("Team Goal Left Post")
    right_post = pos("Team Goal Right Post")
    opp_left_post = pos("Opponent Goal Left Post")
    opp_right_post = pos("Opponent Goal Right Post")

    p1 = pos("Team Player 1")
    p2 = pos("Team Player 2")
    p3 = pos("Team Player 3")
    p4 = pos("Team Player 4")
    team = [p1, p2, p3, p4]

    opponents = [
        pos("Opponent Player 1"),
        pos("Opponent Player 2"),
        pos("Opponent Player 3"),
        pos("Opponent Player 4"),
    ]

    team_has = SoccerGetBool("Team Has Ball")
    opp_has = SoccerGetBool("Opponent Has Ball")
    loose = SoccerGetBool("Is Ball Loose")

    # --- Threat / option debug (always on while the ball is held) ---
    #
    # The old green/red cones were rooted at the Ball transform. The moment
    # anyone picks up, that point is a held-ball offset orbiting the carrier,
    # so the cones skew or look like they vanished. Root everything at the
    # carrier BODY and expand to a 2-ply minimax surface instead:
    #   ply 0: direct shot cone + pass to each teammate
    #   ply 1: from each teammate, shot cone + pass to every other teammate
    # Red/Orange = enemy threats at our goal. Green/Cyan = our options at theirs.
    # Loose ball keeps the simple ball-rooted cones (no holder to tree from).
    # Always draw BOTH threat trees (ours + enemy). Possession only changes
    # which ETA path is live for covering — not whether the lines exist.
    always = Bool(True)
    opp_has_bits = [SoccerGetBool(f"Opponent Player {i} Has Ball") for i in range(1, 5)]
    draw_2ply_threat_tree(
        opponent_carrier(opponents, ball),
        opponents,
        opp_has_bits,
        team_goal,
        left_post,
        right_post,
        always,
        "Red",
        "Orange",
        "Titanium.Enemy",
    )
    our_carrier, team_has_bits = team_carrier(team)
    draw_2ply_threat_tree(
        our_carrier,
        team,
        team_has_bits,
        opp_goal,
        opp_left_post,
        opp_right_post,
        always,
        "Green",
        "Light Blue",
        "Titanium.Attack",
    )
    _, loose_enemy_l = post_tangent(ball, opp_left_post, POST_CONTACT_RADIUS)
    _, loose_enemy_r = post_tangent(ball, opp_right_post, POST_CONTACT_RADIUS)
    _, loose_own_l = post_tangent(ball, left_post, POST_CONTACT_RADIUS)
    _, loose_own_r = post_tangent(ball, right_post, POST_CONTACT_RADIUS)
    DebugDrawLine(ball, ConditionalSetVector3(loose, loose_enemy_l, ball), Float(0.06), "Green")
    DebugDrawLine(ball, ConditionalSetVector3(loose, loose_enemy_r, ball), Float(0.06), "Green")
    DebugDrawLine(ball, ConditionalSetVector3(loose, loose_own_l, ball), Float(0.06), "Red")
    DebugDrawLine(ball, ConditionalSetVector3(loose, loose_own_r, ball), Float(0.06), "Red")

    for i, opp in enumerate(opponents, start=1):
        plot_xz(f"Titanium.Opp{i}.Pos", "Red", opp)
    # Diagnostic: does SoccerGetVector3("Ball Velocity") actually read a live
    # value in the real game, or does it collapse to ~0? predict_ball_meet_point
    # degenerates to "just chase the ball's current position" whenever
    # velocity reads near-zero, which would look exactly like "never
    # intercepts, just walks to the ball" regardless of the solve itself.
    plot_xz("Titanium.BallVel", "Magenta", SoccerGetVector3("Ball Velocity"))

    r_int = SoccerGetFloat("Player Interact Radius")
    r_eff = r_int + Float(BALL_RADIUS)

    ball_vel = SoccerGetVector3("Ball Velocity")
    # --- Outfield players 1-3: multi-body goalkeeper ---
    #
    # The team defends as one keeper with four bodies. When the opponent has
    # the ball, each outfielder seals a DISTINCT opponent's scoring cone
    # (`threat_cover`, the same construction P4 uses on the carrier), so every
    # player who could receive a pass and shoot already has their lane shut
    # rather than the keeper guessing which pass is coming. Assignment is by
    # index so the three cover three different opponents — the fourth is the
    # carrier, whom the keeper is pressing directly.
    #
    # With the ball, the same players push forward and take the shot the tick
    # a lane opens (`build_carrier_move`), sitting on a permanently full
    # charge until then.
    # Attacking shape, ball-relative. `fwd` is the attack axis and `lat` is
    # its left-hand perpendicular in the pitch plane, so the stations rotate
    # with the direction of play instead of being pinned to world axes.
    #
    # Every outfielder previously computed the SAME support target and they
    # duly walked into one pile — four bodies inside a few metres, nobody
    # anywhere else to pass to, attack stalled with the ball stuck in a
    # corner. Distinct stations are what stop that: two wide either side and
    # ahead of the ball to stretch the defence, one trailing goal-side as the
    # safe backward outlet (which is also the receiver the carrier will look
    # for when it cannot escape a tackle).
    # Enemy 2-ply clocks + seal points, ordered most→least immediate for the
    # three outfielders (rank 0/1/2). P4 still presses the carrier directly.
    threat_times, threat_covers = threat_times_and_covers(
        opponents,
        opp_has_bits,
        opponent_carrier(opponents, ball),
        team_goal,
        left_post,
        right_post,
        r_int,
        our_carrier,
        team_has,
        opp_has,
        ball,
        loose,
    )
    # Which cones are already sealed by somebody (incl. P4) — spare outfielders
    # are then free to intercept instead of double-marking.
    threat_sealed = [
        threat_already_sealed(team, opp, team_goal, left_post, right_post, r_int)
        for opp in opponents
    ]

    fwd = unit_or_zero(opp_goal - ball)
    lat = Vector3(Float(0) - fwd.z, Float(0), fwd.x)
    # P3 is the lackey: glued behind the carrier for a guaranteed escape pass.
    lackey_pos = lackey_station(our_carrier, opp_goal)

    # Flanker/press role: only a body that can WIN the stamina duel is sent.
    # Soft P3 by default; if they're spent, closest capable outfielder presses
    # and P3 inherits that player's cover urgency (clean swap).
    opp_carrier_body = opponent_carrier(opponents, ball)
    opp_carrier_stam = SoccerGetFloat("Ball Carrier Stamina")
    press_flags, cover_urgencies, any_presser = outfield_press_roles(
        [p1, p2, p3], opp_carrier_stam, opp_carrier_body
    )

    for slot, me, ahead, side in (
        (1, p1, 9.0, 7.0),
        (2, p2, 9.0, -7.0),
        (3, p3, -6.0, 0.0),
    ):
        has = SoccerGetBool(f"Team Player {slot} Has Ball")
        charge = SoccerGetFloat(f"Teammate {slot} Shot Charge")
        closest = SoccerGetBool(f"Is Team Player {slot} Closest Teammate to Ball")
        urgency = cover_urgencies[slot - 1]
        i_am_presser = press_flags[slot - 1]

        carry, shoot_now = build_carrier_move(
            me,
            ball,
            opp_goal,
            opp_left_post,
            opp_right_post,
            team,
            opponents,
            r_eff,
            r_int,
            has,
            charge,
        )
        cover_move, on_cover = cover_or_intercept(
            urgency,
            threat_times,
            threat_covers,
            threat_sealed,
            me,
            ball,
            ball_vel,
        )
        # Free defenders (all threats sealed, or this slot has no open mark)
        # chase the ball; otherwise hold the urgency-ranked seal.
        intercept_free = predict_ball_meet_point(me, ball, ball_vel, SPRINT_SPEED)
        defend = ConditionalSetVector3(on_cover, cover_move, intercept_free)
        # Capable presser: go win the stall; seal only on a cheap detour.
        # Incapable flanker never gets this job (stamina gate above).
        aggro = aggressive_press_with_cover(
            me, cover_move, on_cover, opp_carrier_body, r_int
        )
        defend = ConditionalSetVector3(
            And(And(opp_has, i_am_presser), any_presser), aggro, defend
        )
        support = ball + fwd * Float(ahead) + lat * Float(side)
        # While WE hold: same cover clocks are live (steal→shot), not cosmetics.
        # If any opponent can actually win the stamina duel, seal those cones;
        # if nobody can take the ball, push into attack stations instead.
        any_steal = Bool(False)
        for t_steal in threat_times:
            any_steal = Or(any_steal, CompareFloats(t_steal, Float(500.0), "<"))
        with_ball_offball = ConditionalSetVector3(any_steal, defend, support)
        # Lackey (P3): stay in the carrier's back pocket whenever we possess,
        # so there is always one guaranteed safe-pass body if anti-tackle fails.
        if slot == 3:
            with_ball_offball = ConditionalSetVector3(team_has, lackey_pos, with_ball_offball)
        move = ConditionalSetVector3(And(team_has, Not(has)), with_ball_offball, defend)
        # When defending (opp has / loose threats), use cover-or-intercept —
        # not the attacking support station.
        move = ConditionalSetVector3(And(Not(team_has), Not(has)), defend, move)
        move = ConditionalSetVector3(has, carry, move)

        # Loose ball is nobody's yet. Two different situations, not one:
        #   - our own pass in flight toward this player: no opponent is
        #     realistically closer, so this is a controlled reception, not a
        #     scramble. Walk to meet the ball's predicted path instead of
        #     chasing its live position (which lags a moving ball and either
        #     stalls the receiver in place or has him trailing it).
        #   - a genuine 50/50 (opponent clearance, deflection, loose after a
        #     tackle): an opponent is comparably close, so this needs the
        #     sprint to actually win the race.
        # Stamina IS ball retention (tackler_stam >= carrier_stam decides
        # every duel, and sprinting is the only drain), so spending it on an
        # uncontested reception is pure waste — exactly what let a fresher
        # opponent tackle us straight back after a hard-won interception.
        # Contested was inverted: it's the actual race, so it needs the lead
        # point most, at the speed we'll really be moving (sprint, since
        # that's what `sprint` below sets for this branch) — chasing live
        # position here was bleeding ground to the ball in exactly the case
        # where winning the race matters most.
        meet_point_walk = predict_ball_meet_point(me, ball, ball_vel, WALK_SPEED)
        meet_point_sprint = predict_ball_meet_point(me, ball, ball_vel, SPRINT_SPEED)
        contested = CompareFloats(nearest_opponent_dist(opponents, ball), Distance(me, ball) * Float(1.3), "<=")
        loose_target = ConditionalSetVector3(contested, meet_point_sprint, meet_point_walk)
        move = ConditionalSetVector3(And(loose, closest), loose_target, move)
        # Sprint: contested loose, spare free defender, or stamina-capable
        # presser closing on the opponent carrier.
        press_sprint = And(
            And(And(opp_has, Not(has)), Not(team_has)),
            And(i_am_presser, any_presser),
        )
        sprint = Or(
            And(loose, And(closest, contested)),
            Or(
                And(And(Not(on_cover), Not(has)), Not(team_has)),
                press_sprint,
            ),
        )
        plot_xz(f"Titanium.P{slot}.Pos", "Cyan", me)
        plot_xz(f"Titanium.P{slot}.Target", "Yellow", move)
        DebugDrawLine(me, move, Float(0.05), "White")
        DebugDrawDisc(move, Float(0.2), Float(0.05), "Yellow")
        SoccerController(slot, move, sprint, player_interact(slot, has, shoot_now))

    # --- Player 4 goalkeeper ---
    h4 = SoccerGetBool("Team Player 4 Has Ball")
    c4 = SoccerGetFloat("Teammate 4 Shot Charge")
    carrier_charge = SoccerGetFloat("Ball Carrier Shot Charge")
    move4, sprint4, interact4 = gk_policy(
        p4,
        ball,
        ball_vel,
        team_goal,
        left_post,
        right_post,
        opponents,
        r_eff,
        h4,
        c4,
        opp_has,
        loose,
        carrier_charge,
    )
    closest4 = SoccerGetBool("Is Team Player 4 Closest Teammate to Ball")
    move4 = ConditionalSetVector3(And(loose, closest4), ball, move4)
    # Re-apply champion half clamp after the loose-ball override (same as
    # gk_policy) so chasing a far loose ball cannot yank P4 over midfield.
    move4 = ConditionalSetVector3(h4, move4, clamp_own_half(move4, team_goal))
    plot_xz("Titanium.P4.Pos", "Cyan", p4)
    plot_xz("Titanium.P4.Target", "Yellow", move4)
    DebugDrawLine(p4, move4, Float(0.05), "White")
    DebugDrawDisc(move4, Float(0.2), Float(0.05), "Yellow")
    SoccerController(4, move4, sprint4, interact4)

    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SaveData(str(CANDIDATE_OUT), layout="grid")
    print(f"Wrote candidate: {CANDIDATE_OUT}")
    print(f"nodes={len(graph_data['serializableNodes'])} conns={len(graph_data['serializableConnections'])}")

    # Always refresh Titanium_test for side-by-side playtesting. Never touches
    # live Titanium.txt unless --promote is passed after the gate wins.
    text = CANDIDATE_OUT.read_text(encoding="utf-8")
    TEST_SAVES.parent.mkdir(parents=True, exist_ok=True)
    TEST_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    TEST_SAVES.write_text(text, encoding="utf-8")
    TEST_LOCAL.write_text(text, encoding="utf-8")
    print(f"Wrote playtest: {TEST_SAVES}")
    print(f"Wrote playtest: {TEST_LOCAL}")

    if "--promote" not in sys.argv:
        print(
            "\nNOT deployed to live Titanium.txt — only candidate + Titanium_test.\n"
            "Gate it first (aicomp-soccer-sim/scripts/gate_round_robin.py) "
            "against the currently-live build, then re-run with --promote "
            "once it's actually won."
        )
        return

    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_OUT.is_file():
        prior = LOCAL_OUT.read_bytes()
        n = 0
        backup_path = BACKUPS_DIR / "Titanium_pre_promote.txt"
        while backup_path.is_file() and backup_path.read_bytes() != prior:
            n += 1
            backup_path = BACKUPS_DIR / f"Titanium_pre_promote_{n}.txt"
        if not backup_path.is_file():
            backup_path.write_bytes(prior)
        print(f"Backed up previously-live build to {backup_path}")

    SAVES.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAVES.write_text(text, encoding="utf-8")
    LOCAL_OUT.write_text(text, encoding="utf-8")
    print(f"PROMOTED to live: {SAVES}")
    print(f"PROMOTED to live: {LOCAL_OUT}")


if __name__ == "__main__":
    main()
