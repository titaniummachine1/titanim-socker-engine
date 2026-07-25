"""Where to walk to meet a moving ball — the single interception solve.

Its own module on purpose. This lived inside `ball_physics` (539 lines), and
splicing a replacement into that file corrupted it repeatedly; as one file
with one entry point it can be swapped wholesale instead of edited in place.

Samples the SHARED PerfectController trajectory (`worldcupteams/ball_prediction`)
— the from-scratch port of the simulator's own free-ball model, verified
against it to 3.8-5.8 cm. Deliberately NOT `ball_trajectory_graph._event_leg`,
whose chain carries the bugs that port exists to replace (airborne legs
ignoring walls, `terminal` meaning the post-bounce coast point, tangential
velocity surviving a wall stop).
"""
from __future__ import annotations

import os
import sys

_WORLDCUPTEAMS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "worldcupteams"
)
for _p in (_WORLDCUPTEAMS, os.path.join(_WORLDCUPTEAMS, "AIGamePyLibrary")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from titanium._env import *  # noqa: F401,F403,E402

import ball_prediction as bp  # noqa: E402

# Refinements of the pursuit fixed point. Three is what the previous solve
# used and is plenty: each pass only corrects the travel-time estimate.
REFINEMENTS = 3


def _clamp01(x):
    x = ConditionalSetFloat(CompareFloats(x, bp._f(1.0), ">"), bp._f(1.0), x)
    return ConditionalSetFloat(CompareFloats(x, bp._f(0.0), "<"), bp._f(0.0), x)


def _build_legs(px, pz, vx, vz, live_y):
    """Run the trajectory once and keep each leg's (start, end, duration).

    Built ONCE per call site; the pursuit refinements then sample it, so three
    iterations cost three cheap lookups rather than three whole solves.
    """
    B = bp.pitch_bounds()
    dead = CompareFloats(bp._f(0.0), bp._f(1.0), ">")  # constant false
    # Loose ball: grounded and rolling from here. No ball-height signal exists
    # to justify an airborne guess, and guessing wrong over-throws the target.
    air = bp._f(0.0)

    legs = []
    for _ in range(bp.LEGS):
        sx, sz = px, pz
        px, pz, vx, vz, air, dead, dur = bp._leg(px, pz, vx, vz, air, dead, B)
        legs.append(((sx, sz), (px, pz), dur))
    return legs


def _position_at(legs, t, live_y):
    """Ball position `t` seconds along the predicted path.

    Walks the legs forward accumulating durations. Each leg's fraction clamps
    to [0,1], so a leg already finished contributes its END point and a leg
    not yet reached contributes nothing — the last leg whose start time has
    passed therefore wins, which is exactly the containing leg. Position is
    interpolated linearly within a leg, matching how PerfectController draws
    its own breakpoints.
    """
    (sx0, sz0), _, _ = legs[0]
    point = Vector3(sx0, live_y, sz0)
    elapsed = bp._f(0.0)
    for (sx, sz), (ex, ez), dur in legs:
        frac = _clamp01(bp._safe_div(t - elapsed, dur))
        here = Vector3(sx + (ex - sx) * frac, live_y, sz + (ez - sz) * frac)
        point = ConditionalSetVector3(
            CompareFloats(t, elapsed, ">="), here, point
        )
        elapsed = elapsed + dur
    return point


SAMPLES = 12
"""Trajectory samples used to find the earliest reachable point. Twelve over
the whole path is fine resolution: the ball is slowest (so samples are
densest in space) exactly where the decision is tight."""


def predict_ball_meet_point(me, ball, ball_vel, speed):
    """EARLIEST point on the ball's path this player can actually reach.

    This is the interception question: "what is the first moment I can be
    where the ball is?" — not "where will the ball be when I arrive?", and
    emphatically not "where does the ball stop". The resting point is only a
    terminal condition, used when no earlier point is reachable at all.

    Walk sampled times along the predicted path and keep the earliest `t`
    where `distance(me, ball_at(t)) <= speed * t` — the player can be there
    by then. Selection runs last-to-first so an earlier reachable sample
    always overrides a later one.

    Two previous formulations both lost matches despite being more accurate:

      * `ball + velocity * t`, unbounded — targets 250 m off the pitch, so
        the chaser trailed the ball forever (measured P1.Target = 252, 141).
      * A fixed-point pursuit on the exact trajectory — converged to the
        ball's REST position (accurate to 4 cm) and gated 6:10 vs 16:4,
        because walking to where the ball stops means arriving late. Accuracy
        was never the problem; answering the wrong question was.
    """
    parts = Vector3Split(ball)
    vparts = Vector3Split(ball_vel)
    live_y = parts.y

    legs = _build_legs(parts.x, parts.z, vparts.x, vparts.z, live_y)
    total = bp._f(0.0)
    for _, _, dur in legs:
        total = total + dur

    # Fallback: the ball's resting point, i.e. nothing earlier was reachable.
    terminal = _position_at(legs, total, live_y)
    point = terminal

    # Last-to-first so the EARLIEST reachable sample wins.
    for i in reversed(range(1, SAMPLES + 1)):
        t = total * bp._f(float(i) / SAMPLES)
        p = _position_at(legs, t, live_y)
        reachable = CompareFloats(Distance(me, p), Float(speed) * t, "<=")
        point = ConditionalSetVector3(reachable, p, point)
    return point
