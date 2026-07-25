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
HOLD_PROXY = 0.55


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


_EVENT_LEG_BUILT = False


def _build_event_leg_function():
    """`_event_leg` (ball_trajectory_graph's per-segment friction+wall-bounce
    solve) built ONCE as a real reusable graph function instead of being
    inlined at every call site — the whole point being fewer total nodes.

    It needs six outputs; graph functions here only return one value, so
    the rest publish through SetVariable/GetVariable, the same side-channel
    ball_trajectory_graph itself already uses for its per-segment outputs.
    The one value actually returned (`goal_point`) is always consumed by
    every caller, so the call can never be pruned as dead/unused code by
    the demand-driven interpreter — that would silently drop the
    SetVariable side effects the other five outputs depend on.
    """
    global _EVENT_LEG_BUILT
    if _EVENT_LEG_BUILT:
        return
    traj.build_trajectory_physics_functions()
    fn = traj._begin_function("TitaniumEventLeg")
    # Function params are typed "Any" in this library, so `.x`/`.z` (which
    # gate on self.type == "Vector3") raise on them directly — go through
    # Vector3Split (a plain node wiring, no such type gate) and reconstruct
    # proper Vector3 nodes before handing them to `_event_leg`, which does
    # use `.x`/`.z` internally.
    start_parts = Vector3Split(fn.Param1)
    velocity_parts = Vector3Split(fn.Param2)
    start = Vector3(start_parts.x, start_parts.y, start_parts.z)
    velocity = Vector3(velocity_parts.x, Float(0), velocity_parts.z)
    live_y = fn.Param3
    leg = traj._event_leg(start, velocity, live_y)
    SetVariable("_TitaniumEventLegHasGoal", leg["has_goal"])
    SetVariable("_TitaniumEventLegHasWall", leg["has_wall"])
    SetVariable("_TitaniumEventLegEventTime", leg["event_time"])
    SetVariable("_TitaniumEventLegContact", leg["contact"])
    SetVariable("_TitaniumEventLegBounceVelocity", leg["bounce_velocity"])
    traj._finish(
        fn,
        leg["goal_point"],
        (fn.Param1, fn.Param2, fn.Param3, start, velocity, leg["goal_point"]),
    )
    _EVENT_LEG_BUILT = True


@cache
def _call_event_leg(start, velocity, live_y):
    """Call the shared TitaniumEventLeg function; its five side-channel
    outputs must be read out immediately (before this is called again for
    another leg in the same tick) since they publish through a single
    shared variable slot, not a per-call one — same constraint
    ball_trajectory_graph's own segment chaining has."""
    _build_event_leg_function()
    goal_point = CustomFunction("TitaniumEventLeg", start, velocity, live_y)
    return {
        "has_goal": GetVariable("_TitaniumEventLegHasGoal"),
        "has_wall": GetVariable("_TitaniumEventLegHasWall"),
        "event_time": GetVariable("_TitaniumEventLegEventTime"),
        "goal_point": goal_point,
        "contact": GetVariable("_TitaniumEventLegContact"),
        "bounce_velocity": GetVariable("_TitaniumEventLegBounceVelocity"),
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

    # ONE grounded segment only. A shot that bounces off a wall and then goes
    # in is not a threat anyone can actually aim, so chasing that case is pure
    # wasted node budget — the second `_event_leg` call is dropped. (Wall
    # bounces are still modelled correctly by ball_trajectory_graph; we just
    # do not predict *goals* through them.)
    leg1 = _call_event_leg(landing_point, velocity, live_y)
    # CustomFunction/GetVariable results come back typed "Any" in this
    # library, so `.x` (which gates on self.type == "Vector3") would raise —
    # Vector3Split has no such gate.
    goal1_x = Vector3Split(leg1["goal_point"]).x
    at_own_1 = CompareFloats(Abs(goal1_x - own_goal_x), Float(0.5), "<=")
    ground_threat = And(leg1["has_goal"], at_own_1)
    ground_time = hang_time + leg1["event_time"]
    ground_point = leg1["goal_point"]

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
    dir_l, _ = post_tangent(shot_origin, opp_left_post, POST_CONTACT_RADIUS)
    dir_r, _ = post_tangent(shot_origin, opp_right_post, POST_CONTACT_RADIUS)
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


def build_carrier_move(me, ball, opp_goal, opp_left_post, opp_right_post, opponents, r_eff, has_ball, charge):
    """With ball: walk a safe lane toward goal sitting on a full charge, and
    on the tick a scoring lane opens snap MoveTo onto the aim direction.

    The snap *is* the aim — the engine kicks along MoveTo — so the release
    and the snap have to land on the same tick. That is why `shoot_now` is
    returned rather than recomputed: it must drive Interact too.

    The `charge` gate is not optional. Releasing means dropping Interact, and
    Interact is also what *builds* the charge — so firing the moment a lane
    happens to be open (which, on pickup in space, is immediately) holds the
    ball at zero charge forever: the engine needs >0.05 to kick at all, so
    nothing is ever struck. Wait for the charge to actually be banked, then
    spend it.
    """
    # Shot origin is the CARRIER, not the ball's transform: the real engine
    # visually orbits the held ball at an offset in front of the player, but
    # the actual kick fires from the player's own center. Aiming the tangent
    # geometry from `ball` (the offset position) instead of `me` was solving
    # a slightly wrong triangle every time.
    lane_ok, aim = clear_shot(me, opp_goal, opp_left_post, opp_right_post, opponents, r_eff)
    ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    shoot_now = And(has_ball, And(lane_ok, ready))
    walk = safe_walk_target(me, ball, opp_goal, opponents, r_eff)
    shoot_to = me + aim * Float(10)
    move = ConditionalSetVector3(shoot_now, shoot_to, walk)
    # Without ball: chase ball with same cone (avoid running through tacklers).
    chase = safe_walk_target(me, ball, ball, opponents, r_eff, step=8.0)
    move = ConditionalSetVector3(has_ball, move, chase)
    return move, shoot_now


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

    Targeting the ball's current position always lags a moving ball — by the
    time you arrive, it has moved on, which either stalls you dead in its
    path waiting for it to arrive or, worse, has you retreating to chase it
    as it passes. This is a fixed-point pursuit solve (3 refinements): guess
    a travel time from the straight-line distance, project the ball forward
    that long, then re-measure distance to the projected point and repeat.

    Deliberately a straight-line extrapolation (no friction decel) — passes
    are short-range and this is meant to get a receiver moving promptly at
    a controlled walk, not to nail the exact final rest point.
    """
    velocity = Vector3(ball_vel.x, Float(0), ball_vel.z)
    point = ball
    for _ in range(3):
        t = Distance(me, point) / Float(speed)
        point = ball + velocity * t
    return point


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

    # The keeper never crosses midfield, however tempting the press or a
    # loose ball upfield looks — being beaten past the halfway line leaves an
    # open net behind him, and the outfield cover players exist to handle
    # anything out there.
    #
    # Deliberately NOT applied while carrying: the kick fires along MoveTo, so
    # clamping x would rotate the clear aim back toward our own goal.
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
        "Titanium",
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

    # Debug: true (post-tangent corrected) min/max straight-line shot cone
    # from the ball, at both goals. Green = cone into the enemy goal (what
    # we could score with right now); red = cone into our own goal (the
    # danger cone the GK must seal). Uses the existing DebugDrawLine /
    # DebugDrawDisc nodes AIA exposes — no custom node types added.
    _, enemy_aim_l = post_tangent(ball, opp_left_post, POST_CONTACT_RADIUS)
    _, enemy_aim_r = post_tangent(ball, opp_right_post, POST_CONTACT_RADIUS)
    DebugDrawLine(ball, enemy_aim_l, Float(0.06), "Green")
    DebugDrawLine(ball, enemy_aim_r, Float(0.06), "Green")
    DebugDrawDisc(enemy_aim_l, Float(0.15), Float(0.05), "Green")
    DebugDrawDisc(enemy_aim_r, Float(0.15), Float(0.05), "Green")

    _, own_aim_l = post_tangent(ball, left_post, POST_CONTACT_RADIUS)
    _, own_aim_r = post_tangent(ball, right_post, POST_CONTACT_RADIUS)
    DebugDrawLine(ball, own_aim_l, Float(0.06), "Red")
    DebugDrawLine(ball, own_aim_r, Float(0.06), "Red")
    DebugDrawDisc(own_aim_l, Float(0.15), Float(0.05), "Red")
    DebugDrawDisc(own_aim_r, Float(0.15), Float(0.05), "Red")

    p1 = pos("Team Player 1")
    p2 = pos("Team Player 2")
    p3 = pos("Team Player 3")
    p4 = pos("Team Player 4")

    opponents = [
        pos("Opponent Player 1"),
        pos("Opponent Player 2"),
        pos("Opponent Player 3"),
        pos("Opponent Player 4"),
    ]
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
    team_has = SoccerGetBool("Team Has Ball")
    opp_has = SoccerGetBool("Opponent Has Ball")
    loose = SoccerGetBool("Is Ball Loose")

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
    fwd = unit_or_zero(opp_goal - ball)
    lat = Vector3(Float(0) - fwd.z, Float(0), fwd.x)

    for slot, me, marks, ahead, side in (
        (1, p1, opponents[0], 9.0, 7.0),
        (2, p2, opponents[1], 9.0, -7.0),
        (3, p3, opponents[2], -6.0, 0.0),
    ):
        has = SoccerGetBool(f"Team Player {slot} Has Ball")
        charge = SoccerGetFloat(f"Teammate {slot} Shot Charge")
        closest = SoccerGetBool(f"Is Team Player {slot} Closest Teammate to Ball")

        carry, shoot_now = build_carrier_move(
            me, ball, opp_goal, opp_left_post, opp_right_post, opponents, r_eff, has, charge
        )
        # Defensive default: seal our assigned opponent's shot cone.
        move = threat_cover(marks, team_goal, left_post, right_post, r_int)
        # Attacking: hold this player's own station in the shape.
        support = ball + fwd * Float(ahead) + lat * Float(side)
        move = ConditionalSetVector3(And(team_has, Not(has)), support, move)
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
        sprint = And(loose, And(closest, contested))
        plot_xz(f"Titanium.P{slot}.Pos", "Cyan", me)
        plot_xz(f"Titanium.P{slot}.Target", "Yellow", move)
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
    # Loose ball: go to ball only if closest teammate; sprint already gated
    # inside gk_policy (goal-bound-shot walk-vs-sprint check).
    closest4 = SoccerGetBool("Is Team Player 4 Closest Teammate to Ball")
    move4 = ConditionalSetVector3(And(loose, closest4), ball, move4)
    plot_xz("Titanium.P4.Pos", "Cyan", p4)
    plot_xz("Titanium.P4.Target", "Yellow", move4)
    SoccerController(4, move4, sprint4, interact4)

    SAVES.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SaveData(str(SAVES), layout="grid")
    # Also write a local copy for the sim repo.
    Path(SAVES).replace  # noqa: keep linter calm
    text = SAVES.read_text(encoding="utf-8")
    LOCAL_OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {SAVES}")
    print(f"Wrote {LOCAL_OUT}")
    print(f"nodes={len(graph_data['serializableNodes'])} conns={len(graph_data['serializableConnections'])}")


if __name__ == "__main__":
    main()
