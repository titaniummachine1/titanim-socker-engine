"""Ball trajectory prediction: given a start position and velocity, where
does the ball actually go?

This is the "simulate the ball's path, given where a player kicked it from
and which direction/strength, so we know where it lands and whether it
bounces" module — airborne glide, then up to two grounded friction+wall-
bounce legs, using `ball_trajectory_graph`'s shared, calibrated physics
(`traj._event_leg`) rather than reimplementing it.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium._env import traj
from titanium.constants import (
    GRAVITY,
    KICK_LIFT_BASE,
    KICK_LIFT_PER_CHARGE,
    KICK_SPEED_BASE,
    KICK_SPEED_PER_CHARGE,
)
from titanium.geometry import clamp01, unit_or_zero

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
    clobbers before the first is read (this caused a 0-10 loss previously;
    verified again via probes/build_function_nesting_probe.py).

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
def call_event_leg_chain(start, velocity, live_y, goal_x):
    """Call the shared TitaniumEventLegChain function (up to 2 wall
    bounces). Its two side-channel outputs (`is_threat`, `time`) must be
    read immediately after this call, before any other call to this
    function in the same tick — same single-shared-variable-slot
    constraint documented on `_build_event_leg_chain_function`."""
    _build_event_leg_chain_function()
    point = CustomFunction("TitaniumEventLegChain", start, velocity, live_y, goal_x)
    return {
        "is_threat": GetVariable("_TitaniumEventLegChainIsThreat"),
        "time": GetVariable("_TitaniumEventLegChainTime"),
        "point": point,
    }


def airborne_hang_time(speed):
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

    KNOWN LIMITATION: this is a rough same-tick estimate re-derived from
    CURRENT speed every tick, not a fixed value anchored to the actual kick
    moment. As the ball decelerates mid-flight this re-guess shifts — a
    real sensing gap (no ball-height signal exists), not a bug in this
    function. A proper fix would anchor hang-time to the actual kick-
    release event and count down from there instead of re-estimating.

    NOTE: a "just drop this, assume grounded" rewrite was tried and tested
    WORSE against the live champion (missed a genuine airborne-crossing
    shot the champion caught) — the estimate is flawed but not net-harmful
    versus having no airborne detection at all. Do not remove without a
    gate-verified replacement.
    """
    charge_est = clamp01((speed - Float(KICK_SPEED_BASE)) / Float(KICK_SPEED_PER_CHARGE))
    lift_raw = Float(KICK_LIFT_BASE) + Float(KICK_LIFT_PER_CHARGE) * charge_est
    lift_est = ConditionalSetFloat(CompareFloats(lift_raw, Float(0), "<"), Float(0), lift_raw)
    return (Float(2.0) * lift_est) / Float(GRAVITY)


@cache
def own_goal_threat(ball, ball_vel, own_goal_x, goal_half_width):
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
    hang_time = airborne_hang_time(speed)

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
    chain = call_event_leg_chain(landing_point, velocity, live_y, own_goal_x)
    ground_threat = chain["is_threat"]
    ground_time = hang_time + chain["time"]
    ground_point = chain["point"]

    is_threat = Or(airborne_threat, ground_threat)
    time = ConditionalSetFloat(airborne_threat, t_cross, ground_time)
    point = ConditionalSetVector3(airborne_threat, airborne_point, ground_point)
    return is_threat, time, point


def trajectory_breakpoints(origin, velocity, hang_time, live_y):
    """Shared leg-chaining: given a KNOWN origin/velocity/hang_time, glide
    then up to 2 grounded friction+bounce legs. Used both for a loose
    ball's predicted path (hang_time = 0, assumed already grounded — see
    `own_goal_threat`) and a hypothetical kick's predicted path (hang_time
    EXACT, from a known charge, see `kick_launch_from_charge`) — the
    leg-chaining physics past the airborne phase is identical either way,
    only how `hang_time`/`velocity` were obtained differs.

    Returns `(landing, leg1, leg2, leg3, leg1_end, leg2_start, leg2_end,
    leg3_start, leg3_end)` — the raw per-leg dicts plus the breakpoints a
    drawer needs, each breakpoint collapsing to the previous one once a
    further leg isn't reached (so a caller drawing segments between
    consecutive breakpoints never needs its own has_wall bookkeeping).
    """
    landing = origin + velocity * hang_time
    leg1 = traj._event_leg(landing, velocity, live_y)
    leg2 = traj._event_leg(leg1["contact"], leg1["bounce_velocity"], live_y)
    leg3 = traj._event_leg(leg2["contact"], leg2["bounce_velocity"], live_y)
    # A leg that ends at a wall must be DRAWN to the wall contact. Do not use
    # `leg["terminal"]` here: when has_wall it resolves to `coast_stop`, the
    # rest point *after* the bounce, so drawing to it cuts the corner straight
    # through the wall and then restarts the next segment back at the contact
    # — a stray line and a path that appears to ignore walls entirely.
    # `ball_trajectory_graph._trajectory_solve` already does it this way.
    leg1_end = ConditionalSetVector3(
        leg1["has_goal"],
        leg1["goal_point"],
        ConditionalSetVector3(leg1["has_wall"], leg1["contact"], leg1["friction_stop"]),
    )
    leg2_start = ConditionalSetVector3(leg1["has_wall"], leg1["contact"], leg1_end)
    leg2_end = ConditionalSetVector3(
        leg1["has_wall"],
        ConditionalSetVector3(
            leg2["has_goal"],
            leg2["goal_point"],
            ConditionalSetVector3(
                leg2["has_wall"], leg2["contact"], leg2["friction_stop"]
            ),
        ),
        leg2_start,
    )
    leg3_start = ConditionalSetVector3(
        And(leg1["has_wall"], leg2["has_wall"]), leg2["contact"], leg2_end
    )
    leg3_end = ConditionalSetVector3(
        And(leg1["has_wall"], leg2["has_wall"]),
        ConditionalSetVector3(leg3["has_goal"], leg3["goal_point"], leg3["terminal"]),
        leg3_start,
    )
    return landing, leg1, leg2, leg3, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end


def kick_launch_from_charge(charge):
    """Forward direction: given a KNOWN shot charge (0..1), the exact
    launch speed and airborne hang time the engine's own kick formula
    implies. This is exact — we know our own charge directly, unlike
    trying to back-solve a charge from an already-observed speed for a
    shot we don't control (see `own_goal_threat`'s docstring for why that
    estimate was dropped). Useful for previewing "if I released right now,
    where would this go" before actually releasing.
    """
    c = clamp01(charge)
    speed = Float(KICK_SPEED_BASE) + Float(KICK_SPEED_PER_CHARGE) * c
    lift_raw = Float(KICK_LIFT_BASE) + Float(KICK_LIFT_PER_CHARGE) * c
    lift = ConditionalSetFloat(CompareFloats(lift_raw, Float(0), "<"), Float(0), lift_raw)
    hang_time = (Float(2.0) * lift) / Float(GRAVITY)
    return speed, hang_time


def simulate_kick_breakpoints(origin, aim_dir, charge, live_y):
    """Full predicted flight of a hypothetical kick from `origin` toward
    unit `aim_dir` at `charge` — exact launch speed/hang-time (see
    `kick_launch_from_charge`), then the same leg-chained bounce physics
    as everything else in this module. Returns
    `(hang_time, landing, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end)`.
    """
    speed, hang_time = kick_launch_from_charge(charge)
    velocity = aim_dir * speed
    landing, _leg1, _leg2, _leg3, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end = trajectory_breakpoints(
        origin, velocity, hang_time, live_y
    )
    return hang_time, landing, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end


def _clamp_axis(v, lo, hi):
    v = ConditionalSetFloat(CompareFloats(v, Float(hi), ">"), Float(hi), v)
    return ConditionalSetFloat(CompareFloats(v, Float(lo), "<"), Float(lo), v)


@cache
def predict_ball_meet_point(me, ball, ball_vel, speed):
    """Where to walk to meet a moving ball, instead of its live position.

    Fixed-point pursuit solve (3 refinements): guess a travel time from the
    straight-line distance, project the ball forward that long, re-measure,
    repeat.

    The projection obeys FRICTION and stops where the ball stops. It used to
    be a bare `ball + velocity * t` — no friction, no walls, no bound — which
    at 28 m/s produced targets ~250 m off a +-40 x +-25 pitch (measured:
    P1.Target = 252.1, 141.2, exactly along the velocity ray). The chaser was
    ordered somewhere it could never reach, so it trailed the ball forever
    instead of cutting across to meet it. Fixing this beat the previous
    champion 24:4 head-to-head and took Poponeta from 16:6 to 16:4.

    Distance under constant deceleration is `v*t - a*t^2/2`, capped at the
    stop distance `v^2/(2a)`, then clamped to the pitch. Walls are NOT
    modelled: a bounce only shortens the path, so this stays an upper bound.

    NOT yet the shared PerfectController solve (`worldcupteams/ball_prediction`)
    — that port is still outstanding. An attempt sampling the OLD
    `traj._event_leg` chain gated WORSE (Poponeta 10:6 vs 16:4) because its Z
    axis pinned to a leg boundary, so a half-correct trajectory lost to this
    clean approximation. Re-gate anything that replaces it.
    """
    velocity = Vector3(ball_vel.x, Float(0), ball_vel.z)
    ball_speed = Magnitude(velocity)
    direction = unit_or_zero(velocity)
    a = Float(traj.FRICTION_ACCEL)
    stop_distance = (ball_speed * ball_speed) / (Float(2.0) * a)

    point = ball
    for _ in range(3):
        t = Distance(me, point) / Float(speed)
        travelled = ball_speed * t - Float(0.5) * a * t * t
        travelled = ConditionalSetFloat(
            CompareFloats(travelled, stop_distance, ">"), stop_distance, travelled
        )
        travelled = ConditionalSetFloat(
            CompareFloats(travelled, Float(0.0), "<"), Float(0.0), travelled
        )
        point = ball + direction * travelled

    parts = Vector3Split(point)
    return Vector3(
        _clamp_axis(parts.x, traj.XMIN, traj.XMAX),
        parts.y,
        _clamp_axis(parts.z, traj.ZMIN, traj.ZMAX),
    )
