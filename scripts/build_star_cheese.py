"""Build real-game StarCheese.txt via AIGamePyLibrary.

A team that keeps the ball by making it unreachable, rather than by dribbling
away from pressure. Three ideas, all of them consequences of measured rules:

  * YOU CAN TACKLE YOUR OWN TEAMMATES (confirmed by the game's author), so
    possession can be MOVED without ever kicking. A tackle is instant and the
    ball cannot be intercepted in flight, because there is no flight.
  * A HELD BALL SITS AT `carrier + facing * hold_offset` and facing is the walk
    direction, applied instantly. So the ball is wherever the carrier is
    pointing, and turning relocates it by up to 2 * hold_offset in one tick.
  * THE CARRIER RESOLVES INTERACTION FIRST, then everyone else nearest-ball
    first. A teammate in reach therefore takes the ball on the same tick.

Put together: four players stand one chain-spacing apart, so any of them can
take the ball off any other instantly. The ball lives on whichever player has
the most free directions -- Titanium's blocked-arc model, reused as a target
selector instead of an escape planner. An opponent closing on the ball is
exactly what pushes the ball off it, because approaching widens the arc that
opponent blocks.

PORT NOTES -- read before changing anything
===========================================
This is a BRANCH-FOR-BRANCH port of `aicomp-soccer-sim/src/star_cheese_ref.rs`
(identical logic to `src/bin/star_cheese.rs`, which is the measurement harness).
Every `if` in the reference exists here as a `ConditionalSet*`, including the
ones the flags below currently switch off, so a divergence found by
`star_parity` is a bug in one file or the other and not a missing feature.

Where the reference reads state a node graph CANNOT read, the substitution is
at the INPUT, never in the logic. There are exactly three, and each is either
exact or bounded:

  1. CARRIER VELOCITY -- exact, not an approximation. The reference feeds
     `predict` each player's `p.vel`; no per-player velocity accessor exists.
     But the engine reports a HELD ball's velocity as its carrier's (real
     TimePlot, aicomp-soccer-sim `possession.rs` quirk #16, `ball.vel = p.vel`
     every tick while held), so `Ball Velocity` IS the carrier's velocity --
     ours when we hold it, theirs when they do. The reference's non-lite chase
     reads the opponent carrier's `c.vel` and gets the same number.
  2. NON-CARRIER VELOCITY -- the one genuine approximation. Substituted with
     `unit(to_target) * min(walk_speed, dist/dt)`: zero when a player is
     standing on its slot (so a settled spoke-holder stays put, which is the
     case that matters for holding a chain) and full cruise when it is running.
     Error is bounded by one tick of the accel ramp, ~3 cm.
  3. CARRIER FACING -- exact. `predict` returns the old facing when the move
     target is where the player already stands, and facing is only ever read
     for the carrier (it places the ball). A held ball sits at
     `carrier + facing * hold_offset`, so `unit(ball - carrier)` recovers it
     exactly, except while the hold point is compressed against a wall.

One reference `if` cannot survive at all: `pressed[]`, the press history that
makes Interact a pulse. Variables here are NOT one-tick memory -- the brain
runs 8 settle passes per tick over every SetVariable with vars persisting
across ticks (verified in `graph_vm/runtime_brain.rs`, not taken from a
docstring), so a `B := A; A := expr` delay chain collapses to `B == expr`
inside the same tick. The reference's `want && !pressed` is a 2-tick square
wave while `want` holds, so it is rebuilt from the clock instead:
`round(Current Simulation Time / fixed_dt) % 2`. Same duty cycle, phase set by
absolute tick rather than by when `want` turned on.

Measured in the simulator against the six-opponent roster (180 s, Home side):
86 goals for, 0 against, versus Titanium's 25 for and 75 against on identical
fixtures. See `aicomp-soccer-sim/src/bin/star_cheese.rs` -- the harness is NOT
the AI, it only exists to check the idea was worth building.

    python scripts/build_star_cheese.py               # uncompressed + verifiable
    python scripts/build_star_cheese.py --compress    # only after parity passes
    python scripts/build_star_cheese.py --lite        # graph-feasible variant
    python scripts/build_star_cheese.py --promote
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from titanium._env import *  # noqa: F401,F403,E402
from titanium._env import graph_data  # noqa: E402
from titanium.geometry import pos, unit_or_zero  # noqa: E402
from titanium.nodefn import graph_function  # noqa: E402
from titanium.paths import SAVES  # noqa: E402

# ---------------------------------------------------------------------------
# Flags. Same names and same defaults as the Rust harness CLI, so a fixture can
# be reproduced on both sides. `star_parity` compares against
# `decide(.., lite=false, nearest_goal=false)` with intercept ON, which is what
# these defaults build.
# ---------------------------------------------------------------------------

# GRAPH-FEASIBLE mode (`--lite` in star_cheese.rs): acos-free freedom proxy,
# 4 fixed spokes instead of a 16-way greedy search, 12 sampled intercept times
# instead of 160. Kept because it is the cheap variant, NOT because the full
# one is impossible -- both are built here.
# SHIPPING DEFAULT IS THE SMALL BUILD. `--full` unrolls the interception
# search to 160 one-tick samples so it matches `star_cheese_ref` sample for
# sample, which is what `star_parity` needs to attribute a divergence. That
# costs 10,000 nodes and 26 MB and MEASURES WORSE: 75-4 across the roster
# against the 12-sample build's 79-3.
#
# Extra samples buy precision on a decision that is already coarse — a player
# either gets to the ball or does not — so the fine sweep only adds ways to
# pick a marginally different interception point. Debug fidelity is worth
# paying for while debugging, not while playing.
FULL = "--full" in sys.argv
LITE = not FULL
# Compression is `optimize="release"` (DCE + debug-strip). It is nearly free
# here — 11323 -> 11263 nodes — because the size is the unroll, not the debug
# sinks. Kept on by default anyway; `--no-compress` keeps every Debug/TimePlot
# sink so a `star_parity` divergence names something still in the file.
COMPRESS = "--no-compress" not in sys.argv
# Mirrors the reference harness's flags so parity runs compare like with like.
# Interception on: chase where the ball WILL be, not where it is (+31 goals).
INTERCEPT_ON = "--no-intercept" not in sys.argv
# Nearest-goal aim OFF: it measured worse (45-19 against 44-16 alone, and it
# cost 3 goals on top of interception). The centre spot is not a detour — it is
# the one place the chain can be fed from every angle.
NEAREST_GOAL = "--nearest" in sys.argv

# ---------------------------------------------------------------------------
# Constants. Every one of these is measured; none is a tuning knob. Values that
# the game exposes through an accessor are read from the game instead (see
# `_read_world`), because a hardcoded copy is a second place to be wrong.
# ---------------------------------------------------------------------------

# Held-ball offset along facing. Real-game TimePlot, 2026-07-26.
HOLD_OFFSET = 1.67
# Cruise speed. Measured 2026-07-25.
WALK_SPEED = 7.0
# Coulomb slide deceleration for a free ball.
SLIDE_ACCEL = 5.95
# Engine fixed step (~52.6 Hz).
FIXED_DT = 0.019
# Player accel/decel, from `SimParams::default` (accel from rest median n=277,
# decel to rest median n=279). These are what make `predict`'s brake branch
# fire: at cruise the brake distance is 49/(2*200) = 12 cm, which is exactly
# where a player holding a spoke sits.
PLAYER_ACCEL = 100.0
PLAYER_DECEL = 200.0
# Chain spacing sits 1 cm INSIDE maximum reach. At exactly hold + interact
# radius the float gap lands on 1.7500001 and the engine's `dist <= radius`
# test refuses, so a held ball at nominal maximum reach is untouchable.
CHAIN_MARGIN = 0.01
# Nominal interact radius, for the static kickoff spots only. Everything in
# play reads `Player Interact Radius` from the game.
R_INT_NOMINAL = 1.75
# Interception horizon. A node graph has no loops: a back-edge is inlined until
# the depth cap and yields null rather than iterating, so the search is written
# out. The reference walks 160 ticks at stride 1 and falls back to the sample
# at 160*dt; --lite takes 12 samples at stride 10 but keeps the SAME 160*dt
# fallback, which looks like an inconsistency and is faithfully reproduced.
INTERCEPT_SAMPLES = 12 if LITE else 160
INTERCEPT_STRIDE = 10 if LITE else 1
INTERCEPT_FALLBACK_TICKS = 160
# Star spokes for --lite, degrees off the carrier-to-goal line, one per slot.
SPOKE_DEG = (0.0, 50.0, -50.0, 110.0)
# Full mode: 16 candidate directions, greedily assigned one per player.
SPOKE_DIRS = 16
# Relay hysteresis and forward-progress margin, from the reference's `better`.
FREER_MARGIN = 0.02
FORWARD_MARGIN = 0.10
# Spoke scoring weights (reference: `1.5 * dot - 2.0 * exposure - 0.05 * run`).
W_FORWARD = 1.5
W_EXPOSURE = 2.0
W_TRAVEL = 0.05
TAU = 2.0 * math.pi

# World reads, published here so graph-function bodies can reference them as
# globals. They MUST already exist in global scope before any body is emitted,
# or `nodefn` would assign them to that function and they would stop evaluating
# outside a call -- hence `_read_world()` runs first and this dict is only ever
# read, never built, inside a body.
_W: dict = {}


def _vlen(v):
    return Magnitude(v)


def _fmin(a, b):
    """`a.min(b)` on floats, as one conditional."""
    return ConditionalSetFloat(CompareFloats(a, b, "<"), a, b)


def _safe(x, floor=1e-4):
    """`x.max(floor)` — guards a divisor the reference also guards."""
    return ConditionalSetFloat(CompareFloats(x, Float(floor), "<"), Float(floor), x)


# ---------------------------------------------------------------------------
# FREEDOM. Reference: `blocked_fraction` / `blocked_fraction_lite` / `freedom`.
#
# Both live behind one `freedom(...)` switch, exactly as in the reference. Each
# per-opponent contribution is a graph FUNCTION: the shape is instantiated for
# 16 candidate spokes x 4 players x 4 opponents plus every predicted end state,
# and a `CreateFunction` body is emitted once while costing the same per-tick
# evaluations as the inlined copies (see titanium/nodefn.py).
# ---------------------------------------------------------------------------


@graph_function("ScArcBlocked", ("Vector3", "Float", "Vector3", "Float"), "Float")
def _arc_blocked(me, my_stam, opp, opp_stam):
    """Radians of this player's ball-circle that ONE opponent denies.

    A held ball traces a circle of radius `hold` around its carrier, so an
    opponent at distance `d` denies an ARC of that circle with half-width

        acos((hold^2 + d^2 - r_int^2) / (2 * hold * d))

    Beyond `hold + r_int` it denies nothing; inside `r_int - hold` it denies
    everything. That, not distance to the nearest opponent, is what "safe"
    means here: a player with an opponent 2 m away on one side is free to turn
    the other way, while a player ringed at 3 m has nowhere to go.

    The reference RETURNS 1.0 early for the swallowed case. A graph cannot
    return early, so it contributes a full turn instead and the caller's
    `ClampFloat(.., 0, 1)` produces the identical 1.0 -- the only place in this
    file where "same branch" and "same statement" differ.
    """
    r_int = _W["r_int"]
    d = _vlen(opp - me)
    # A weaker opponent loses the stamina duel, so it is not a threat at all.
    capable = CompareFloats(opp_stam, my_stam, ">=")
    far = CompareFloats(d, Float(HOLD_OFFSET) + r_int, ">")
    swallowed = CompareFloats(d + Float(HOLD_OFFSET), r_int, "<=")
    denom = MultiplyFloats(Float(2.0 * HOLD_OFFSET), d)
    cos_half = ClampFloat(
        (Float(HOLD_OFFSET * HOLD_OFFSET) + MultiplyFloats(d, d) - MultiplyFloats(r_int, r_int))
        / _safe(denom),
        Float(-1),
        Float(1),
    )
    arc = MultiplyFloats(Float(2), Acos(cos_half))
    contrib = ConditionalSetFloat(
        far, Float(0), ConditionalSetFloat(swallowed, Float(TAU), arc)
    )
    return ConditionalSetFloat(capable, contrib, Float(0))


@graph_function("ScArcBlockedLite", ("Vector3", "Float", "Vector3", "Float"), "Float")
def _arc_blocked_lite(me, my_stam, opp, opp_stam):
    """Acos-free proxy: how deep this opponent has pushed into the annulus.

    Same ordering as the arc integral without the acos -- zero outside
    hold + r_int, growing as it closes, and monotone in the arc it denies,
    which is all a "who is freest" comparison needs.
    """
    r_int = _W["r_int"]
    d = _vlen(opp - me)
    capable = CompareFloats(opp_stam, my_stam, ">=")
    depth = (Float(HOLD_OFFSET) + r_int) - d
    contrib = ConditionalSetFloat(CompareFloats(depth, Float(0), ">"), depth, Float(0))
    return ConditionalSetFloat(capable, contrib, Float(0))


def freedom(me, my_stam):
    """Fraction of `me`'s ball-circle that opponents deny, in [0, 1].

    0.0 means every direction is safe -- this player cannot be tackled no
    matter which way it turns. 1.0 means it is dead whatever it does.

    Arcs are summed, not merged, so overlapping opponents over-count. That is
    deliberate: over-counting is conservative, it only ever makes a covered
    player look worse, and merging is not needed to pick the best of four.
    """
    contrib = _arc_blocked_lite if LITE else _arc_blocked
    total = Float(0)
    for opp, opp_stam in zip(_W["threats"], _W["opp_stam"]):
        total = total + contrib(me, my_stam, opp, opp_stam)
    # Reference divides by TAU (full) or 4*(hold + r_int) (lite) then `.min(1)`.
    # Both numerators are >= 0, so clamping the low end changes nothing and the
    # native ClampFloat is one node where a second conditional would be two.
    if LITE:
        scale = MultiplyFloats(Float(4), Float(HOLD_OFFSET) + _W["r_int"])
    else:
        scale = Float(TAU)
    return ClampFloat(total / scale, Float(0), Float(1))


# ---------------------------------------------------------------------------
# PREDICT. Reference: `predict` — mirrors the engine's `step_mover` exactly.
# ---------------------------------------------------------------------------


def predict(here, vel, move_to, facing, dt=FIXED_DT):
    """Where a player ENDS this tick, and which way it will be facing.

    Interaction is resolved AFTER movement, so every reach test and every hold
    point has to be evaluated here, not at the position the player is standing
    on while the decision is being made. Rotate to the move direction
    instantly, then brake-to-rest or accelerate-and-step.

    Measured: in the simulator harness this single correction took the design
    from 0 goals in every fixture to 44. It is not a refinement.
    """
    decel = max(PLAYER_DECEL, 1e-6)
    to = move_to - here
    dist = _vlen(to)
    speed = _vlen(vel)
    moving = CompareFloats(speed, Float(1e-6), ">")
    brake = ConditionalSetFloat(
        moving, MultiplyFloats(speed, speed) / Float(2.0 * decel), Float(0)
    )
    turning = CompareFloats(dist, Float(1e-6), ">")
    new_facing = ConditionalSetVector3(turning, unit_or_zero(to), facing)

    # ---- inside the braking envelope: coast to rest along the CURRENT
    # velocity, which is not necessarily the direction of the new target.
    dv = Float(decel * dt)
    brake_step = ConditionalSetFloat(
        CompareFloats(speed, dv, "<="),
        MultiplyFloats(speed, speed) / Float(2.0 * decel),
        MultiplyFloats(speed, Float(dt)) - Float(0.5 * decel * dt * dt),
    )
    coasting = here + ScaleVector3(unit_or_zero(vel), brake_step)
    arriving = ConditionalSetVector3(moving, coasting, here)

    # ---- otherwise: ramp toward cruise at accel, clamp, and step.
    max_speed = Float(WALK_SPEED)
    desired = ScaleVector3(new_facing, max_speed)
    delta = desired - vel
    max_delta = Float(PLAYER_ACCEL * dt)
    v = ConditionalSetVector3(
        CompareFloats(_vlen(delta), max_delta, "<="),
        desired,
        vel + ScaleVector3(unit_or_zero(delta), max_delta),
    )
    v_len = _vlen(v)
    v = ConditionalSetVector3(
        CompareFloats(v_len, max_speed, ">"),
        ScaleVector3(v, max_speed / _safe(v_len, 1e-6)),
        v,
    )
    stepping = here + ScaleVector3(v, Float(dt))

    at_target = Or(
        CompareFloats(dist, Float(1e-3), "<="), CompareFloats(dist, brake, "<=")
    )
    return ConditionalSetVector3(at_target, arriving, stepping), new_facing


# ---------------------------------------------------------------------------
# BALL PATH + INTERCEPTION. Reference: `ball_at` / `intercept_point`.
# ---------------------------------------------------------------------------


def ball_at(ball, bvel, t, decel):
    """Where the ball is after `t` seconds.

    Two cases, one formula. A LOOSE ball is a Coulomb slide: constant
    deceleration along its heading, stopping dead at zero. A HELD ball rides its
    carrier, keeping that velocity with no slide. `Ball Velocity` reports the
    carrier's velocity while held, so the same read serves both.
    """
    speed = _vlen(bvel)
    dirv = unit_or_zero(bvel)
    still = CompareFloats(speed, Float(1e-4), "<")
    if decel <= 1e-4:
        return ConditionalSetVector3(still, ball, ball + ScaleVector3(bvel, Float(t)))
    stop_t = speed / Float(decel)
    tt = _fmin(Float(t), stop_t)
    travel = MultiplyFloats(speed, tt) - MultiplyFloats(
        Float(0.5 * decel), MultiplyFloats(tt, tt)
    )
    return ConditionalSetVector3(still, ball, ball + ScaleVector3(dirv, travel))


def clamp_to_pitch(v):
    """Keep a target on the field.

    An interception solve extrapolates up to ~3 s ahead, so a ball moving at
    walk speed projects ~21 m forward — routinely past the goal line. Chasing a
    point off the pitch walks the whole team off the pitch, which is how a side
    concedes without ever contesting anything.

    Reference clamps `x` to +-x_max and `y` to +-z_max; pitch `y` is our `z`,
    x_max is half the Field DEPTH and z_max half the Field WIDTH.
    """
    return Vector3(
        ClampFloat(v.x, Float(0) - _W["x_max"], _W["x_max"]),
        Float(0),
        ClampFloat(v.z, Float(0) - _W["z_max"], _W["z_max"]),
    )


def intercept_point(me, bvel, decel):
    """Earliest point on the ball's path this player can actually reach.

    Walking at where the ball IS is walking at where it has already left, which
    never closes on anything moving away. This takes the first sampled moment
    the player can be within interact range of where the ball WILL be — a
    cut-off rather than a pursuit.

    Unrolled LATEST-FIRST so the ConditionalSetVector3 chain leaves the
    EARLIEST satisfied sample on top, which is what the reference's `return`
    inside a forward loop does.
    """
    ball = _W["ball"]
    r_int = _W["r_int"]
    best = clamp_to_pitch(
        ball_at(ball, bvel, INTERCEPT_FALLBACK_TICKS * FIXED_DT, decel)
    )
    # Reference: `for k in 0..n` — k never reaches n.
    for k in range(INTERCEPT_SAMPLES - 1, -1, -1):
        t = k * INTERCEPT_STRIDE * FIXED_DT
        bp = ball_at(ball, bvel, t, decel)
        # Where we can have got to by then, plus the reach we arrive with.
        reachable = CompareFloats(_vlen(bp - me), Float(WALK_SPEED * t) + r_int, "<=")
        best = ConditionalSetVector3(reachable, clamp_to_pitch(bp), best)
    return best


# ---------------------------------------------------------------------------
# STAR. Reference: the spoke loop inside `decide`.
# ---------------------------------------------------------------------------


def spoke_lite(slot, anchor, to_goal, spacing):
    """Fixed spokes: straight on, and +/-50, +110 degrees.

    Reference builds this from `atan2(to_goal) + off`; rotating `to_goal` by the
    same angle in the x/z plane is the identical vector without an atan2 node.
    """
    lat = Vector3(Float(0) - to_goal.z, Float(0), to_goal.x)
    a = math.radians(SPOKE_DEG[slot])
    d = ScaleVector3(to_goal, Float(math.cos(a))) + ScaleVector3(lat, Float(math.sin(a)))
    return anchor + ScaleVector3(unit_or_zero(d), spacing)


def spoke_greedy(slot, anchor, to_goal, spacing, searching, chosen_before):
    """One direction each out of 16, greedily assigned in slot order.

    Scores forward progress against how exposed the spot is, minus how far this
    player has to run to get there (so spokes do not swap every tick). Simple
    and good enough to choose among 16.

    `chosen_before` is the list of `(searching_q, chose_k_q)` for every EARLIER
    slot: the reference's `taken` vector, which only earlier players that
    actually SEARCHED contribute to — a carrier or a chaser `continue`s before
    it can claim a direction. Returns `(spot, chose_k)` where `chose_k` is the
    16 one-hot booleans the next slot needs.
    """
    me = _W["mates"][slot]
    my_stam = _W["mate_stam"][slot]
    best_score = Float(-1e9)  # reference seeds with f32::NEG_INFINITY
    best_spot = anchor
    best_idx = Float(-1)
    any_valid = Bool(False)

    for k in range(SPOKE_DIRS):
        a = k * TAU / SPOKE_DIRS
        d = Vector3(Float(math.cos(a)), Float(0), Float(math.sin(a)))
        spot = anchor + ScaleVector3(d, spacing)
        # Keep spokes on the pitch.
        on_pitch = And(
            CompareFloats(Abs(spot.x), _W["x_max"] - Float(1), "<="),
            CompareFloats(Abs(spot.z), _W["z_max"] - Float(1), "<="),
        )
        taken = Bool(False)
        for prev_searching, prev_chose in chosen_before:
            taken = Or(taken, And(prev_searching, prev_chose[k]))
        usable = And(on_pitch, Not(taken))

        exposure = freedom(spot, my_stam)
        score = (
            MultiplyFloats(Float(W_FORWARD), DotProduct(d, to_goal))
            - MultiplyFloats(Float(W_EXPOSURE), exposure)
            - MultiplyFloats(Float(W_TRAVEL), _vlen(spot - me))
        )
        better = And(usable, CompareFloats(score, best_score, ">"))
        best_score = ConditionalSetFloat(better, score, best_score)
        best_spot = ConditionalSetVector3(better, spot, best_spot)
        best_idx = ConditionalSetFloat(better, Float(k), best_idx)
        any_valid = Or(any_valid, usable)

    # `if best.2 != usize::MAX { .. } else { anchor }`
    spot = ConditionalSetVector3(any_valid, best_spot, anchor)
    chose = [
        And(any_valid, CompareFloats(best_idx, Float(k), "=="))
        for k in range(SPOKE_DIRS)
    ]
    return spot, chose


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------


def _read_world() -> None:
    """Every game read, once, into `_W`.

    Called before anything else so that no `SoccerGet*` node is first created
    inside a graph-function body (`nodefn` would assign it to that function and
    it would stop evaluating for every other reader).
    """
    _W["ball"] = pos("Ball")
    _W["ball_vel"] = SoccerGetVector3("Ball Velocity")
    _W["r_int"] = SoccerGetFloat("Player Interact Radius")
    _W["opp_goal"] = pos("Opponent Goal Center")
    # Pitch half-extents, read from the game rather than hardcoded. Reference
    # `x_max` is the goal axis (Field Depth 80 -> 40), `z_max` the sideline
    # axis (Field Width 50 -> 25).
    _W["x_max"] = SoccerGetFloat("Field Depth") / Float(2)
    _W["z_max"] = SoccerGetFloat("Field Width") / Float(2)
    _W["mouth"] = SoccerGetFloat("Goal Width") / Float(2) - Float(0.7)

    _W["mates"] = [pos(f"Team Player {i}") for i in range(1, 5)]
    _W["mate_stam"] = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 5)]
    _W["opp_stam"] = [SoccerGetFloat(f"Opponent Player {i} Stamina") for i in range(1, 5)]
    _W["has"] = [SoccerGetBool(f"Team Player {i} Has Ball") for i in range(1, 5)]
    _W["team_has"] = SoccerGetBool("Team Has Ball")
    _W["loose"] = SoccerGetBool("Is Ball Loose")
    _W["opp_has"] = SoccerGetBool("Opponent Has Ball")

    # Opponents, stepped one tick toward the ball at walk speed — the same
    # prediction the reference makes, and the right one because interaction is
    # resolved after movement.
    opps = [pos(f"Opponent Player {i}") for i in range(1, 5)]
    _W["threats"] = [
        o + ScaleVector3(unit_or_zero(_W["ball"] - o), Float(WALK_SPEED * FIXED_DT))
        for o in opps
    ]

    # Attack direction, so "further forward" survives the away side. The
    # reference only ever runs Home (+x is forward) and compares raw x; this is
    # that comparison written so a sign flip cannot invert it.
    _W["atk"] = Sign(_W["opp_goal"].x)


def _goal_for(here):
    """Reference `goal_for`: the centre spot, or the nearest point on the mouth.

    Aim 2 m PAST the line: the player is clamped to the pitch at x_max, but its
    held ball sits hold_offset further on and is not clamped inside the mouth,
    so walking into the line carries the ball over it.
    """
    goal_c = Vector3(_W["opp_goal"].x, Float(0), Float(0))
    if not NEAREST_GOAL:
        return goal_c
    mouth = _W["mouth"]
    return Vector3(
        _W["opp_goal"].x + MultiplyFloats(_W["atk"], Float(2.0)),
        Float(0),
        ClampFloat(here.z, Float(0) - mouth, mouth),
    )


def _forward(v):
    """Signed progress up the pitch — reference's bare `.x` on the Home side."""
    return MultiplyFloats(v.x, _W["atk"])


def build() -> None:
    # Kickoff shape: the chain, already formed, pointed up the pitch. Starting
    # in formation means the first tick is already a working chain rather than
    # four players walking into one. Static spots, so the nominal interact
    # radius is used here and the live accessor everywhere in play.
    tm = ConditionalSetFloat(SoccerGetBool("Is Home Team"), Float(-1), Float(1))
    ko_spacing = HOLD_OFFSET + R_INT_NOMINAL - CHAIN_MARGIN

    def spot(x, z):
        return ScaleVector3(Vector3(Float(x), Float(0), Float(z)), tm)

    ours = SoccerGetBool("Is Team Kicking off")

    def faceoff(i):
        # Kicking: stand on the ball and string the chain back INTO OUR OWN
        # HALF. Receiving: the engine pushes anyone inside the centre circle out
        # to 7.75, so start there already and keep the chain intact through the
        # whistle instead of being scattered by the push.
        #
        # THE SIGN MATTERS AND IT IS NOT COSMETIC. Declared spots are WORLD
        # ABSOLUTE (`team_space_to_world` is the identity), and a team may only
        # place in its own half: anything past halfway is pulled back to the
        # line EXACTLY, not rejected. This chain used to run the other way, into
        # the opponent's half, so all four spots clamped to x = 0 and the whole
        # team spawned stacked on the centre spot — where, being teammates
        # inside interact radius of each other, they tackled one another to zero
        # stamina within two ticks and then lost every duel by rule.
        #
        # MEASURED, that one sign: 0-23 against Poponeta. `spot()` already
        # applies the home/away flip, so this stays positive here.
        return ConditionalSetVector3(
            ours, spot(i * ko_spacing, 0.0), spot(7.75 + i * ko_spacing, 0.0)
        )

    InitializeSoccer(
        "StarCheese", "Poland", faceoff(0), faceoff(1), faceoff(2), faceoff(3)
    )

    _read_world()
    ball = _W["ball"]
    r_int = _W["r_int"]
    mates = _W["mates"]
    mate_stam = _W["mate_stam"]
    has = _W["has"]
    team_has = _W["team_has"]
    loose = _W["loose"]
    spacing = Float(HOLD_OFFSET) + r_int - Float(CHAIN_MARGIN)

    # ---- ANCHOR: whoever holds the ball, else the ball itself --------------
    anchor = ball
    for i in (3, 2, 1, 0):
        anchor = ConditionalSetVector3(has[i], mates[i], anchor)

    # ---- FREEDOM OF THE CURRENT HOLDER, for the relay comparison ----------
    # Reference measures this at the holder's CURRENT position (not its
    # predicted one) and substitutes 1.0 when the ball is loose or theirs, so
    # anything of ours is an improvement.
    holder_now = Float(1)
    for i in (3, 2, 1, 0):
        holder_now = ConditionalSetFloat(has[i], freedom(mates[i], mate_stam[i]), holder_now)
    ball_blocked = ConditionalSetFloat(team_has, holder_now, Float(1))

    # ---- CHASE: if we do not hold the ball, nothing else matters ----------
    # Loose -> intercept the sliding ball. Enemy-held -> intercept the ball
    # travelling with its carrier, which is what a tackle has to reach. The
    # reference reads the opponent carrier's own velocity; a held ball's
    # reported velocity IS its carrier's, so this is the same number.
    chase_active = Bool(False) if not INTERCEPT_ON else Not(team_has)
    to_goal = unit_or_zero(_goal_for(anchor) - anchor)

    # ---- PASS 1: where is everyone going this tick? -----------------------
    moves = []
    chosen_before: list = []
    for slot in range(4):
        me = mates[slot]
        i_am_carrier = has[slot]
        searching = And(Not(i_am_carrier), Not(chase_active))

        if LITE:
            spoke = spoke_lite(slot, anchor, to_goal, spacing)
        else:
            spoke, chose = spoke_greedy(
                slot, anchor, to_goal, spacing, searching, chosen_before
            )
            chosen_before.append((searching, chose))

        chase = ConditionalSetVector3(
            loose,
            intercept_point(me, _W["ball_vel"], SLIDE_ACCEL),
            intercept_point(me, _W["ball_vel"], 0.0),
        )
        # The carrier drives the whole formation at the goal; teammates hold the
        # star; with no ball, everyone goes to cut it off.
        moves.append(
            ConditionalSetVector3(
                i_am_carrier,
                _goal_for(me),
                ConditionalSetVector3(chase_active, chase, spoke),
            )
        )

    # ---- PREDICT: where everyone ENDS this tick ---------------------------
    ends, faces = [], []
    for slot in range(4):
        me = mates[slot]
        to = moves[slot] - me
        dist = _vlen(to)
        # Substituted inputs; see PORT NOTES. The carrier's velocity and facing
        # are exact reads, a non-carrier's velocity is the cruise estimate.
        cruise = ScaleVector3(
            unit_or_zero(to), _fmin(Float(WALK_SPEED), dist / Float(FIXED_DT))
        )
        vel = ConditionalSetVector3(has[slot], _W["ball_vel"], cruise)
        facing_now = ConditionalSetVector3(
            has[slot], unit_or_zero(ball - me), unit_or_zero(to)
        )
        end, face = predict(me, vel, moves[slot], facing_now)
        ends.append(end)
        faces.append(face)

    # The ball ends at the carrier's PREDICTED hold point: it rotates with the
    # carrier's walk direction and travels with it, all before any tackle.
    carrier_end = ends[3]
    carrier_face = faces[3]
    for i in (2, 1, 0):
        carrier_end = ConditionalSetVector3(has[i], ends[i], carrier_end)
        carrier_face = ConditionalSetVector3(has[i], faces[i], carrier_face)
    held_ball_end = carrier_end + ScaleVector3(carrier_face, Float(HOLD_OFFSET))
    ball_end = ConditionalSetVector3(team_has, held_ball_end, ball)

    # ---- RELAY: the ball goes to the freest player in reach ---------------
    # Reach and freedom are both measured at the predicted end state.
    end_freedom = [freedom(ends[s], mate_stam[s]) for s in range(4)]
    best_f = Float(1e9)
    best_id = Float(0)
    for slot in range(4):
        f = end_freedom[slot]
        in_reach = CompareFloats(_vlen(ends[slot] - ball_end), r_int, "<=")
        # Take it if clearly freer, or as free and further forward.
        clearly_freer = CompareFloats(f, ball_blocked - Float(FREER_MARGIN), "<")
        as_free = CompareFloats(f, ball_blocked + Float(FREER_MARGIN), "<=")
        ahead = CompareFloats(
            _forward(ends[slot]) + Float(HOLD_OFFSET),
            _forward(ball_end) + Float(FORWARD_MARGIN),
            ">",
        )
        better = Or(clearly_freer, And(as_free, ahead))
        candidate = And(And(Not(has[slot]), in_reach), better)
        # `best_taker.is_none_or(|(bf, _)| f < bf)` — strictly freer than the
        # best candidate so far, so an earlier slot wins a tie.
        wins = And(candidate, CompareFloats(f, best_f, "<"))
        best_f = ConditionalSetFloat(wins, f, best_f)
        best_id = ConditionalSetFloat(wins, Float(slot + 1), best_id)

    # ---- PASS 2: who presses interact? ------------------------------------
    # Interact is an IMPULSE: it fires on the RISING edge and needs a release
    # before it can fire again. The reference alternates with `!pressed`; a
    # graph has no one-tick memory, so the same 2-tick square wave comes off
    # the clock. See PORT NOTES.
    tick = Round(SoccerGetFloat("Current Simulation Time") / Float(FIXED_DT))
    armed = CompareFloats(Modulo(tick, Float(2)), Float(0), "==")

    for slot in range(4):
        i_am_carrier = has[slot]
        # Loose ball: reference tests the predicted position against the ball's
        # CURRENT position, which is where a loose ball's `ball_end` sits too.
        # ASK WHO HAS THE BALL FIRST. There are two different jobs here and
        # they were collapsed into one:
        #
        #   WE hold it   -> relay: only the chosen freest player takes it, so
        #                   the team does not shuffle it around pointlessly.
        #   WE DO NOT    -> get it: anyone in reach presses, whether it is
        #                   loose or on an opponent. Reaching an enemy-held
        #                   ball IS the tackle.
        #
        # Without the second branch an enemy-held ball fell through to the
        # relay selector, which ranks OUR players against OUR carrier — a
        # comparison with no meaning when the opponent has the ball. MEASURED:
        # 529 ticks spent inside interact radius of an enemy-held ball across
        # one match and ZERO tackles attempted, losing 0-23.
        want_get = CompareFloats(_vlen(ends[slot] - ball_end), r_int, "<=")
        want_relay = CompareFloats(best_id, Float(slot + 1), "==")
        want = ConditionalSetBool(team_has, want_relay, want_get)

        # RETAIN: a carrier kicks on the FALLING edge of Interact, so holding it
        # every tick means the ball is never kicked and never loose. That is why
        # this team never gives the ball away — it does not shoot at all. The
        # carrier is exempt from the pulse for exactly that reason: it is not
        # trying to fire an impulse, it is refusing to release one.
        interact = Or(i_am_carrier, And(want, armed))

        SoccerController(slot + 1, moves[slot], Bool(False), interact)


def main() -> int:
    build()
    from AIGamePyLibrary import SaveData

    variant = "lite" if LITE else "full"
    # The shipping name is the small build; the debug unroll gets its own file.
    out_name = "StarCheese.txt" if LITE else "StarCheese_full.txt"
    out = Path(__file__).resolve().parent.parent / "out" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    # `layout="grid"`, never "auto": autoLayout resolves every connection with a
    # linear node scan, which at this size (11k nodes, 24k connections) is
    # ~10^9 comparisons and gets the builder OOM-killed. Layout is cosmetic —
    # titanium.deploy uses grid for the same reason.
    if COMPRESS:
        SaveData(str(out), layout="grid", optimize="release")
    else:
        # Uncompressed: keep every node, every Debug/TimePlot sink, and every
        # variable, so `star_parity`'s divergence dump names something that is
        # still in the file.
        SaveData(str(out), layout="grid", optimize="normal", pruneUnusedNodes=False)
    n = len(graph_data["serializableNodes"])
    c = len(graph_data["serializableConnections"])
    mode = "release" if COMPRESS else "UNCOMPRESSED"
    print(f"StarCheese[{variant}, {mode}]: {n} nodes, {c} connections -> {out}")
    print(f"  file {out.stat().st_size / 1e6:.1f} MB")

    if "--promote" in sys.argv:
        # SAVES points at Titanium.txt itself, not the folder.
        dest = SAVES.parent / out_name
        dest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Deployed to game: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
