"""T-formation positioning and coverage-based passing.

When the team has the ball, outfielders form a T around the carrier:
  - Carrier in center with the ball
  - Left support: 3 * r_int to the carrier's left
  - Right support: 3 * r_int to the carrier's right
  - Rear guard: 4 * r_int behind the carrier

Coverage model: a player is "covered" when an opponent is within 2 * r_int
(the pass-danger threshold). Coverage ranges from 0 (no opponent nearby) to
1 (opponent at 0 distance). A player is "open" when no opponent is within
2 * r_int.

Pass logic:
  1. If carrier is open (not covered), keep the ball.
  2. If carrier becomes covered, pass to the furthest-forward teammate who
     is still open and can receive without interception.
  3. If no open teammate, pass to the teammate closest to the enemy goal
     who can receive without interception.
  4. If carrier is less covered than all teammates, keep the ball.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import HOLD_OFFSET
from titanium.geometry import unit_or_zero
from titanium.shot import clear_pass


def t_formation_stations(carrier, opp_goal, r_int):
    """T-formation stations relative to the carrier.

    Returns (left, right, rear) positions:
      - left/right: scale from 1*r_int (at opp goal) to 4*r_int (at half
        pitch / 40 units away), capped at 4*r_int. Near the goal the
        flankers collapse in tight (tackle distance); at midfield they
        spread wide for passing lanes.
      - rear:  4 * r_int behind the carrier (toward own goal)

    The formation rotates with the attack direction, so "left" and "right"
    are always relative to the line from carrier to opponent goal.
    """
    fwd = unit_or_zero(opp_goal - carrier)
    lat = Vector3(Float(0) - fwd.z, Float(0), fwd.x)

    dist_to_goal = Distance(carrier, opp_goal)
    half_pitch = Float(40.0)
    frac = ClampFloat(dist_to_goal / half_pitch, Float(0.0), Float(1.0))
    flank_scale = Float(1.0) + Float(3.0) * frac  # 1..4

    left = carrier + lat * (flank_scale * r_int)
    right = carrier + lat * (Float(0) - flank_scale * r_int)
    rear = carrier + fwd * (Float(0) - Float(4.0) * r_int)

    return left, right, rear


def player_coverage(me, opponents, r_int, my_stam=None, opp_staminas=None):
    """How covered is this player? Returns a float 0..1.

    0 = no opponent within 2 * r_int (fully open)
    1 = opponent at distance 0 (fully covered)

    Coverage = clamp(1 - dist / (2 * r_int), 0, 1) for the closest
    stam-capable opponent. Low-stam opponents don't count (they can't
    tackle).
    """
    track_r = r_int * Float(2.0)
    worst = Float(0.0)

    stams = opp_staminas
    if stams is None:
        stams = [SoccerGetFloat(f"Opponent Player {i} Stamina") for i in range(1, 5)]
    my = my_stam if my_stam is not None else Float(0)

    for opp, stam in zip(opponents, stams):
        capable = CompareFloats(stam, my, ">=")
        d = Distance(me, opp)
        raw_cov = Float(1.0) - d / track_r
        cov = ClampFloat(raw_cov, Float(0.0), Float(1.0))
        cov_eff = ConditionalSetFloat(capable, cov, Float(0.0))
        worst = ConditionalSetFloat(
            CompareFloats(cov_eff, worst, ">"),
            cov_eff,
            worst,
        )
    return worst


def is_open(me, opponents, r_int, my_stam=None, opp_staminas=None):
    """True if no stam-capable opponent is within 2 * r_int of this player."""
    cov = player_coverage(me, opponents, r_int, my_stam, opp_staminas)
    return CompareFloats(cov, Float(0.0), "<=")


def best_formation_pass(
    carrier,
    teammates,
    opponents,
    opp_goal,
    r_int,
    r_eff,
    my_stam,
    direction_ok=None,
):
    """Coverage-based pass selection following the T-formation rules.

    1. Find all teammates who are open and have a clear pass lane.
    2. Among those, pick the one furthest forward (closest to opp goal).
    3. If none open, pick the teammate closest to opp goal with a clear lane.
    4. If carrier is less covered than all passable teammates, don't pass
       (return can_pass=False).

    Returns (can_pass, aim_dir, mate_pos, mate_coverage).
    """
    opp_staminas = [SoccerGetFloat(f"Opponent Player {i} Stamina") for i in range(1, 5)]

    carrier_cov = player_coverage(carrier, opponents, r_int, my_stam, opp_staminas)

    best_mate = teammates[0]
    best_dir = unit_or_zero(teammates[0] - carrier)
    best_d_to_goal = Float(1e9)
    best_cov = Float(1.0)
    any_ok = Bool(False)
    any_open_ok = Bool(False)

    # First pass: find open teammates with clear pass, pick furthest forward
    open_best_mate = teammates[0]
    open_best_dir = unit_or_zero(teammates[0] - carrier)
    open_best_d = Float(1e9)
    open_best_cov = Float(0.0)

    # Second pass: if no open teammate, find any clear-pass teammate closest to goal
    any_best_mate = teammates[0]
    any_best_dir = unit_or_zero(teammates[0] - carrier)
    any_best_d = Float(1e9)
    any_best_cov = Float(1.0)

    for mate in teammates:
        ok = clear_pass(carrier, mate, opponents, r_eff, direction_ok)
        d_to_goal = Distance(mate, opp_goal)
        mate_cov = player_coverage(mate, opponents, r_int, my_stam, opp_staminas)
        mate_open = CompareFloats(mate_cov, Float(0.0), "<=")

        # Track best open+clear teammate (furthest forward = min dist to goal)
        open_better = And(
            And(ok, mate_open),
            Or(Not(any_open_ok), CompareFloats(d_to_goal, open_best_d, "<")),
        )
        open_best_mate = ConditionalSetVector3(open_better, mate, open_best_mate)
        open_best_dir = ConditionalSetVector3(open_better, unit_or_zero(mate - carrier), open_best_dir)
        open_best_d = ConditionalSetFloat(open_better, d_to_goal, open_best_d)
        open_best_cov = ConditionalSetFloat(open_better, mate_cov, open_best_cov)
        any_open_ok = Or(any_open_ok, And(ok, mate_open))

        # Track best any-clear teammate (closest to goal)
        any_better = And(
            ok,
            Or(Not(any_ok), CompareFloats(d_to_goal, any_best_d, "<")),
        )
        any_best_mate = ConditionalSetVector3(any_better, mate, any_best_mate)
        any_best_dir = ConditionalSetVector3(any_better, unit_or_zero(mate - carrier), any_best_dir)
        any_best_d = ConditionalSetFloat(any_better, d_to_goal, any_best_d)
        any_best_cov = ConditionalSetFloat(any_better, mate_cov, any_best_cov)
        any_ok = Or(any_ok, ok)

    # Prefer open teammates; fall back to any clear-pass teammate
    best_mate = ConditionalSetVector3(any_open_ok, open_best_mate, any_best_mate)
    best_dir = ConditionalSetVector3(any_open_ok, open_best_dir, any_best_dir)
    best_d_to_goal = ConditionalSetFloat(any_open_ok, open_best_d, any_best_d)
    best_cov = ConditionalSetFloat(any_open_ok, open_best_cov, any_best_cov)

    # Pass whenever carrier is covered (coverage > 0) and a clear lane exists.
    # Only keep the ball when carrier is fully open (coverage = 0).
    carrier_covered = CompareFloats(carrier_cov, Float(0.0), ">")
    should_pass = And(any_ok, carrier_covered)

    return should_pass, best_dir, best_mate, best_cov
