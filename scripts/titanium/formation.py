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
from titanium.constants import HOLD_OFFSET, GOAL_HALF_WIDTH
from titanium.geometry import unit_or_zero
from titanium.shot import clear_pass, clear_shot


def t_formation_stations(carrier, opp_goal, r_int, team_goal=None):
    """T-formation stations relative to the carrier.

    Returns (left, right, rear) positions:
      - left/right: scale from 1*r_int (near own goal) to 4*r_int (at half
        pitch / 40 units from own goal), capped at 4*r_int. Deep in our
        own half the flankers collapse in tight (tackle distance); at
        midfield they spread wide for passing lanes. Past midfield stays
        at max spread.
      - rear:  2 * r_int ahead of carrier (second-row attacker, pushing forward)

    The formation rotates with the attack direction, so "left" and "right"
    are always relative to the line from carrier to opponent goal.
    """
    fwd = unit_or_zero(opp_goal - carrier)
    lat = Vector3(Float(0) - fwd.z, Float(0), fwd.x)

    if team_goal is None:
        team_goal = Vector3(Float(0), Float(0), Float(0))

    dist_to_own_goal = Distance(carrier, team_goal)
    half_pitch = Float(40.0)
    frac = ClampFloat(dist_to_own_goal / half_pitch, Float(0.0), Float(1.0))
    flank_scale = Float(1.0) + Float(3.0) * frac  # 1..4

    left = carrier + lat * (flank_scale * r_int)
    right = carrier + lat * (Float(0) - flank_scale * r_int)
    rear = carrier + fwd * (Float(2.0) * r_int)  # push forward, not back

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
    opp_left_post=None,
    opp_right_post=None,
):
    """Coverage-based pass selection: if carrier can't shoot, pass to a
    teammate who can. Falls back to open teammates, then any clear lane.

    Priority:
    1. Teammate with a clear shot lane to goal (furthest forward first).
    2. Open teammate with clear pass lane (furthest forward first).
    3. Any teammate with clear pass lane (closest to goal).

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
    any_shooter_ok = Bool(False)

    # Tier 1: teammate who can shoot
    shoot_best_mate = teammates[0]
    shoot_best_dir = unit_or_zero(teammates[0] - carrier)
    shoot_best_d = Float(1e9)
    shoot_best_cov = Float(1.0)

    # Tier 2: open teammate with clear pass
    open_best_mate = teammates[0]
    open_best_dir = unit_or_zero(teammates[0] - carrier)
    open_best_d = Float(1e9)
    open_best_cov = Float(0.0)

    # Tier 3: any teammate with clear pass
    any_best_mate = teammates[0]
    any_best_dir = unit_or_zero(teammates[0] - carrier)
    any_best_d = Float(1e9)
    any_best_cov = Float(1.0)

    for mate in teammates:
        ok = clear_pass(carrier, mate, opponents, r_eff, direction_ok)
        d_to_goal = Distance(mate, opp_goal)
        mate_cov = player_coverage(mate, opponents, r_int, my_stam, opp_staminas)
        mate_open = CompareFloats(mate_cov, Float(0.0), "<=")

        # Check if this teammate can shoot at goal
        mate_can_shoot = Bool(False)
        if opp_left_post is not None and opp_right_post is not None:
            mate_lane, _ = clear_shot(
                mate, opp_goal, opp_left_post, opp_right_post, opponents, r_eff,
            )
            mate_can_shoot = mate_lane

        # Tier 1: can shoot + clear pass → furthest forward
        shoot_better = And(
            And(ok, mate_can_shoot),
            Or(Not(any_shooter_ok), CompareFloats(d_to_goal, shoot_best_d, "<")),
        )
        shoot_best_mate = ConditionalSetVector3(shoot_better, mate, shoot_best_mate)
        shoot_best_dir = ConditionalSetVector3(shoot_better, unit_or_zero(mate - carrier), shoot_best_dir)
        shoot_best_d = ConditionalSetFloat(shoot_better, d_to_goal, shoot_best_d)
        shoot_best_cov = ConditionalSetFloat(shoot_better, mate_cov, shoot_best_cov)
        any_shooter_ok = Or(any_shooter_ok, And(ok, mate_can_shoot))

        # Tier 2: open + clear pass → furthest forward
        open_better = And(
            And(ok, mate_open),
            Or(Not(any_open_ok), CompareFloats(d_to_goal, open_best_d, "<")),
        )
        open_best_mate = ConditionalSetVector3(open_better, mate, open_best_mate)
        open_best_dir = ConditionalSetVector3(open_better, unit_or_zero(mate - carrier), open_best_dir)
        open_best_d = ConditionalSetFloat(open_better, d_to_goal, open_best_d)
        open_best_cov = ConditionalSetFloat(open_better, mate_cov, open_best_cov)
        any_open_ok = Or(any_open_ok, And(ok, mate_open))

        # Tier 3: any clear pass → closest to goal
        any_better = And(
            ok,
            Or(Not(any_ok), CompareFloats(d_to_goal, any_best_d, "<")),
        )
        any_best_mate = ConditionalSetVector3(any_better, mate, any_best_mate)
        any_best_dir = ConditionalSetVector3(any_better, unit_or_zero(mate - carrier), any_best_dir)
        any_best_d = ConditionalSetFloat(any_better, d_to_goal, any_best_d)
        any_best_cov = ConditionalSetFloat(any_better, mate_cov, any_best_cov)
        any_ok = Or(any_ok, ok)

    # Prefer shooter > open > any
    best_mate = ConditionalSetVector3(any_shooter_ok, shoot_best_mate, any_best_mate)
    best_dir = ConditionalSetVector3(any_shooter_ok, shoot_best_dir, any_best_dir)
    best_d_to_goal = ConditionalSetFloat(any_shooter_ok, shoot_best_d, any_best_d)
    best_cov = ConditionalSetFloat(any_shooter_ok, shoot_best_cov, any_best_cov)

    best_mate = ConditionalSetVector3(any_open_ok, open_best_mate, best_mate)
    best_dir = ConditionalSetVector3(any_open_ok, open_best_dir, best_dir)
    best_d_to_goal = ConditionalSetFloat(any_open_ok, open_best_d, best_d_to_goal)
    best_cov = ConditionalSetFloat(any_open_ok, open_best_cov, best_cov)

    # Pass when carrier is covered and a clear lane exists.
    carrier_covered = CompareFloats(carrier_cov, Float(0.0), ">")
    should_pass = And(any_ok, carrier_covered)

    return should_pass, best_dir, best_mate, best_cov
