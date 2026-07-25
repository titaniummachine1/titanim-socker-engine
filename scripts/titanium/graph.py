"""Assembles every module above into the actual node graph — this is the
only place that calls `InitializeSoccer`/`SoccerController` or decides how
the per-player loop and the goalkeeper wire together. Every other module
stays ignorant of "there are 4 players and a GK"; this is where that shape
lives.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import BALL_RADIUS, SPRINT_SPEED, WALK_SPEED
from titanium.geometry import pos, unit_or_zero
from titanium.shot import nearest_opponent_dist
from titanium.carrier import build_carrier_move
from titanium.ball_physics import predict_ball_meet_point
from titanium.goalkeeper import gk_policy, threat_cover
from titanium.tackle import player_interact
from titanium import debug_viz


def build() -> None:
    # Faceoff (team space: +X attack)
    InitializeSoccer(
        "Titanium",
        "Poland",
        Vector3(Float(8), Float(0), Float(3)),
        Vector3(Float(6), Float(0), Float(-6)),
        Vector3(Float(6), Float(0), Float(6)),
        Vector3(Float(-14), Float(0), Float(0)),
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
