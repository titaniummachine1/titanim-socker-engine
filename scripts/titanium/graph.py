"""Assembles every module above into the actual node graph — this is the
only place that calls `InitializeSoccer`/`SoccerController` or decides how
the per-player loop and the goalkeeper wire together. Every other module
stays ignorant of "there are 4 players and a GK"; this is where that shape
lives.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium._env import WITH_ANTI_TACKLE
from titanium.constants import BALL_RADIUS, SPRINT_SPEED, WALK_SPEED
from titanium.geometry import pos, unit_or_zero
from titanium.shot import nearest_opponent_dist
from titanium.carrier import build_carrier_move
from titanium.ball_physics import predict_ball_meet_point
from titanium.goalkeeper import gk_policy, threat_cover
from titanium.tackle import player_interact
from titanium import debug_viz, positioning
from titanium import support_outlets


# Own goal line. Faceoff coordinates are world absolute, so this is a distance
# from the centre spot, not from our end.
GOAL_LINE_X = 40.0
# Keeper stands this far off its own goal, up the goal-centre→ball line. The
# ball is on the centre spot at a kickoff, so that line is just straight out.
GK_FROM_GOAL = 10.0
# Where the engine pushes a receiving player to: circle 7.25 + measured 0.50
# clearance (Test1/Test2 probes, 2026-07-26). Standing here already means the
# push is a no-op and the shape survives the whistle intact.
CIRCLE_STANDOFF = 7.75
# Wall gap either side of the middle player. Bodies are 0.655 radius, so this
# is a shade over two body widths — a wall, not a pile.
WALL_SPACING = 3.0


def _faceoff_spots():
    """Kickoff spawn spots, arranged by hand in the viewer.

    Measured in the real game 2026-07-26 (Test1/Test2 probes): these are WORLD
    ABSOLUTE coordinates, not our own half's frame. There is no mirroring, so
    the same four numbers played from the other end land in the opponent's
    half, where the engine drags them onto the halfway line — which is exactly
    what the original hand-written spots did to three of our four players.

    So every spot is multiplied by ±1 chosen from `Is Home Team`. One side is
    authored and the other is the same numbers with the sign flipped, which is
    all the mirroring amounts to: the away view is the same pitch rotated 180°,
    not reflected, so X and Z flip together. Home defends −X and takes −1.

    Two shapes, switched on whose kickoff it is.

    OURS: P1 stands on the ball, P2/P3 form a wall either side of it. Standing
    in the circle is legal only for the kicking side, and starting on the spot
    means the first touch is immediate instead of after a walk-in.

    THEIRS: the engine would shove anyone inside the circle out to radius 7.75
    anyway, in whatever direction they happen to lie from the centre. Rather
    than be thrown somewhere arbitrary, P1 is placed ON that radius already, on
    our own side of the circle — same distance the push would have produced,
    but pointing at our goal instead of wherever. P2/P3 keep the wall, set back
    with it. Nobody gets moved, so what is drawn here is what plays.

    The keeper ignores both and sits `GK_FROM_GOAL` up the goal-centre→ball
    line, which at a kickoff is simply straight out from goal.
    """
    tm = ConditionalSetFloat(SoccerGetBool("Is Home Team"), Float(-1), Float(1))

    def spot(x, z):
        return ScaleVector3(Vector3(Float(x), Float(0), Float(z)), tm)

    ours = SoccerGetBool("Is Team Kicking off")

    def wall(z):
        """Same wall slot, on the ball when kicking / on the arc when not."""
        return ConditionalSetVector3(ours, spot(0.0, z), spot(CIRCLE_STANDOFF, z))

    return (
        wall(0.0),
        wall(-WALL_SPACING),
        wall(WALL_SPACING),
        spot(GOAL_LINE_X - GK_FROM_GOAL, 0.0),
    )


def build() -> None:
    InitializeSoccer("Titanium", "Poland", *_faceoff_spots())

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
    team4 = [p1, p2, p3, p4]

    opponents = [
        pos("Opponent Player 1"),
        pos("Opponent Player 2"),
        pos("Opponent Player 3"),
        pos("Opponent Player 4"),
    ]
    for i, opp in enumerate(opponents, start=1):
        debug_viz.plot_xz(f"Titanium.Opp{i}.Pos", "Red", opp)

    debug_viz.draw_shot_cones(ball, team4, opponents, opp_left_post, opp_right_post, left_post, right_post)

    # Diagnostic: does SoccerGetVector3("Ball Velocity") actually read a live
    # value in the real game, or does it collapse to ~0? predict_ball_meet_point
    # degenerates to "just chase the ball's current position" whenever
    # velocity reads near-zero, which would look exactly like "never
    # intercepts, just walks to the ball" regardless of the solve itself.
    debug_viz.plot_xz("Titanium.BallVel", "Magenta", SoccerGetVector3("Ball Velocity"))

    r_int = SoccerGetFloat("Player Interact Radius")
    r_eff = r_int + Float(BALL_RADIUS)

    ball_vel = SoccerGetVector3("Ball Velocity")
    team_has = SoccerGetBool("Team Has Ball")
    opp_has = SoccerGetBool("Opponent Has Ball")
    loose = SoccerGetBool("Is Ball Loose")

    debug_viz.draw_ball_path_and_threats(
        ball, ball_vel, loose, team4, opponents,
        opp_goal, opp_left_post, opp_right_post, team_goal, left_post, right_post,
    )

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
    # Attacking shape: distinct L/R/trail stations so supports do not pile.
    # With AT: stations sit on AT-safe hold rays (red ball dirs) and are
    # pulled in until clear_pass from the carrier — helpers must be where the
    # carrier can actually throw, not where a marker obscures half the cone.
    # Without AT: fixed ball-relative flanks (legacy).
    fwd = unit_or_zero(opp_goal - ball)
    lat = Vector3(Float(0) - fwd.z, Float(0), fwd.x)

    # Carrier body is the hard spacing priority — helpers yield around him.
    h1 = SoccerGetBool("Team Player 1 Has Ball")
    h2 = SoccerGetBool("Team Player 2 Has Ball")
    h3 = SoccerGetBool("Team Player 3 Has Ball")
    carrier = ConditionalSetVector3(
        h1, p1, ConditionalSetVector3(h2, p2, ConditionalSetVector3(h3, p3, p4))
    )

    if WITH_ANTI_TACKLE:
        carrier_stam = SoccerGetFloat("Ball Carrier Stamina")
        left_o, right_o, trail_o = support_outlets.at_safe_flank_stations(
            carrier,
            ball,
            opp_goal,
            team_goal,
            opponents,
            r_int,
            r_eff,
            carrier_stam,
        )
        raw_support = [left_o, right_o, trail_o]
        debug_viz.plot_xz("Titanium.Support.Left", "Yellow", left_o)
        debug_viz.plot_xz("Titanium.Support.Right", "Yellow", right_o)
        debug_viz.plot_xz("Titanium.Support.Trail", "Yellow", trail_o)
    else:
        raw_support = []
        for ahead, side in ((9.0, 7.0), (9.0, -7.0), (-6.0, 0.0)):
            raw_support.append(ball + fwd * Float(ahead) + lat * Float(side))

    outfield = [p1, p2, p3]
    for slot, me, marks, raw in (
        (1, p1, opponents[0], raw_support[0]),
        (2, p2, opponents[1], raw_support[1]),
        (3, p3, opponents[2], raw_support[2]),
    ):
        has = SoccerGetBool(f"Team Player {slot} Has Ball")
        charge = SoccerGetFloat(f"Teammate {slot} Shot Charge")
        closest = SoccerGetBool(f"Is Team Player {slot} Closest Teammate to Ball")

        mates = [p for i, p in enumerate(outfield) if i != (slot - 1)]
        carry, shoot_now = build_carrier_move(
            slot,
            me,
            ball,
            opp_goal,
            team_goal,
            opp_left_post,
            opp_right_post,
            opponents,
            mates,
            r_int,
            r_eff,
            has,
            charge,
        )
        # Defensive default: seal our assigned opponent's shot cone.
        move = threat_cover(marks, team_goal, left_post, right_post, r_int)
        # Attacking: helpers clear ≥r_int from carrier (priority) + mates.
        # Carrier path is never clamped — helpers yield, attacker keeps moving.
        mate_anchors = [p for i, p in enumerate(outfield) if i != (slot - 1)]
        support = positioning.clamp_support_station(
            me, raw, mate_anchors, r_int, priority_anchor=carrier
        )
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
        # Sprint ONLY on a contested loose ball. Removing sprint entirely was
        # tested (2026-07-25) and lost 4:20 to a champion that scored 16:4 --
        # losing the race to a 50/50 concedes possession outright, which costs
        # far more than the stamina saved. Do not "just walk everywhere".
        sprint = And(loose, And(closest, contested))
        debug_viz.plot_xz(f"Titanium.P{slot}.Pos", "Cyan", me)
        debug_viz.plot_xz(f"Titanium.P{slot}.Target", "Yellow", move)
        debug_viz.draw_player_move(me, move)
        SoccerController(slot, move, sprint, player_interact(slot, has, shoot_now))

    # --- Player 4 goalkeeper ---
    h4 = SoccerGetBool("Team Player 4 Has Ball")
    c4 = SoccerGetFloat("Teammate 4 Shot Charge")
    carrier_charge = SoccerGetFloat("Ball Carrier Shot Charge")
    move4, sprint4, interact4, gk_debug = gk_policy(
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
    debug_viz.draw_gk_branches(gk_debug)
    # Loose ball: go to ball only if closest teammate; sprint already gated
    # inside gk_policy (goal-bound-shot walk-vs-sprint check).
    closest4 = SoccerGetBool("Is Team Player 4 Closest Teammate to Ball")
    move4 = ConditionalSetVector3(And(loose, closest4), ball, move4)
    debug_viz.plot_xz("Titanium.P4.Pos", "Cyan", p4)
    debug_viz.plot_xz("Titanium.P4.Target", "Yellow", move4)
    debug_viz.draw_player_move(p4, move4)
    SoccerController(4, move4, sprint4, interact4)
