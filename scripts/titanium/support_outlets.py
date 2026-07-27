"""Attacking support stations on anti-tackle safe headings.

Helpers must stand where the carrier has a clear pass (all opps count for
intercept cones) — not on fixed ball-relative flanks the defence can sit in.
Stations lie on the same rays AT marks safe for the held ball (red discs).
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import HOLD_OFFSET, SUPPORT_OUTLET_DISTS
from titanium.geometry import unit_or_zero
from titanium.shot import clear_pass


def _neg(v):
    return Vector3(Float(0) - v.x, v.y, Float(0) - v.z)


def station_on_heading(carrier, heading, opponents, r_eff, min_dist, direction_ok=None):
    """Farthest point along `heading` with a clear pass from carrier (>= min_dist).

    `direction_ok` rejects headings a defender already owns: a helper standing
    on an angle the carrier cannot even turn to face is not an outlet, because
    the ball is lost the moment the carrier points that way.
    """
    # `clear_pass` tests a DIRECTION cone anchored at `carrier`, and every point
    # on this ray normalises to the same direction — so legality is identical at
    # every distance. The old loop ran the cone solve once per entry in
    # SUPPORT_OUTLET_DISTS and could only ever pick the farthest passing one,
    # which is what this computes directly (952 nodes / 1864 edges cheaper).
    #
    # The reason a distance sweep cannot help here: the opponent model is an
    # infinite cone with no length, so a defender 20 m behind the carrier blocks
    # a 4.5 m outlet exactly as hard as a 9 m one. Giving cones a length is a
    # real change to the model, and would have to clear the gate on its own.
    h = unit_or_zero(heading)
    far = Float(max(SUPPORT_OUTLET_DISTS))
    ok = clear_pass(carrier, carrier + h, opponents, r_eff, direction_ok)
    best = ConditionalSetVector3(ok, carrier + h * far, carrier + h * min_dist)
    return best, ok


def at_safe_flank_stations(
    carrier,
    ball,
    opp_goal,
    team_goal,
    opponents,
    r_int,
    r_eff,
    my_stam,
    danger=None,
    at_debug=None,
):
    """Left / right / trail outlets on AT-safe rays with clear pass from carrier.

    Left/right = max lateral Dot among safe AT probe headings that still admit
    a clear pass. Trail = safest back outlet. Fallbacks are forward-left /
    forward-right / back, still pulled in until clear_pass (or min sep).

    Pass a shared `at_debug` from one `search_safe_direction` to avoid rebuilding
    the entire AT probe set (support + carrier used to pay for it twice).
    """
    from titanium import anti_tackle

    if danger is None:
        danger = Bool(False)

    desired = unit_or_zero(opp_goal - carrier)
    lat = Vector3(Float(0) - desired.z, Float(0), desired.x)
    min_dist = r_int + Float(HOLD_OFFSET)

    aimable = anti_tackle.aim_is_safe(carrier, opponents, r_int, my_stam)
    left_fb, _ = station_on_heading(
        carrier, unit_or_zero(desired + lat), opponents, r_eff, min_dist, aimable
    )
    right_fb, _ = station_on_heading(
        carrier, unit_or_zero(desired + _neg(lat)), opponents, r_eff, min_dist, aimable
    )
    trail_fb, _ = station_on_heading(carrier, _neg(desired), opponents, r_eff, min_dist, aimable)

    if at_debug is None:
        _dir, debug = anti_tackle.search_safe_direction(
            carrier, opp_goal, ball, opponents, r_int, team_goal, my_stam, danger
        )
        _ = _dir
    else:
        debug = at_debug

    left = left_fb
    right = right_fb
    trail = trail_fb
    best_left = Float(-1e9)
    best_right = Float(-1e9)
    best_trail = Float(-1e9)

    for safe, usable, heading, _abs_off, _t, _w, _ev, _prog in debug["probes"]:
        st, clear_ok = station_on_heading(carrier, heading, opponents, r_eff, min_dist, aimable)
        side = DotProduct(lat, heading)
        fwd = DotProduct(desired, heading)
        # Prefer forward-usable AT dirs, then any AT-safe dir.
        q = ConditionalSetFloat(usable, Float(2), ConditionalSetFloat(safe, Float(1), Float(0)))
        base = And(safe, clear_ok)

        left_ok = And(base, CompareFloats(side, Float(0.05), ">"))
        left_score = side + q * Float(0.1)
        better_l = And(left_ok, CompareFloats(left_score, best_left, ">"))
        left = ConditionalSetVector3(better_l, st, left)
        best_left = ConditionalSetFloat(better_l, left_score, best_left)

        right_ok = And(base, CompareFloats(side, Float(-0.05), "<"))
        right_score = (Float(0) - side) + q * Float(0.1)
        better_r = And(right_ok, CompareFloats(right_score, best_right, ">"))
        right = ConditionalSetVector3(better_r, st, right)
        best_right = ConditionalSetFloat(better_r, right_score, best_right)

        trail_ok = And(base, CompareFloats(fwd, Float(-0.05), "<"))
        trail_score = (Float(0) - fwd) + q * Float(0.1)
        better_t = And(trail_ok, CompareFloats(trail_score, best_trail, ">"))
        trail = ConditionalSetVector3(better_t, st, trail)
        best_trail = ConditionalSetFloat(better_t, trail_score, best_trail)

    return left, right, trail
