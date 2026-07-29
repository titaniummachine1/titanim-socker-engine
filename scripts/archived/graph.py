"""Assembles every module above into the actual node graph — this is the
only place that calls `InitializeSoccer`/`SoccerController` or decides how
the per-player loop and the goalkeeper wire together. Every other module
stays ignorant of "there are 4 players and a GK"; this is where that shape
lives.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium._env import WITH_ANTI_TACKLE
from titanium.constants import BALL_RADIUS, CARRIER_SPEED, SPRINT_SPEED, WALK_SPEED
from titanium.geometry import pos, unit_or_zero
from titanium.shot import unopposed_walk_in
from titanium.carrier import build_carrier_move
from titanium.ball_physics import (
    assign_fastest_net_saver,
    own_goal_threat,
    predict_ball_meet_point,
)
from titanium.goalkeeper import gk_policy, threat_cover
from titanium.tackle import held_ball_cutoff, player_interact, tackle_plan
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
    h4_early = SoccerGetBool("Team Player 4 Has Ball")
    carrier = ConditionalSetVector3(
        h1, p1, ConditionalSetVector3(h2, p2, ConditionalSetVector3(h3, p3, p4))
    )

    outfield = [p1, p2, p3]
    all_bodies = [p1, p2, p3, p4]
    has_flags = [h1, h2, h3, h4_early]
    closest_flags = [
        SoccerGetBool(f"Is Team Player {i} Closest Teammate to Ball") for i in range(1, 5)
    ]
    carrier_stam = SoccerGetFloat("Ball Carrier Stamina")
    carrier_charge = SoccerGetFloat("Ball Carrier Shot Charge")
    attack_walk_in_open, attack_walk_in_pt = unopposed_walk_in(
        ball, opp_left_post, opp_right_post, opponents
    )

    # ONE anti-tackle search shared by support stations + carrier walk + urgency.
    # (Previously rebuilt ~5×: support + urgent + each of 3 outfield carriers.)
    if WITH_ANTI_TACKLE:
        from titanium import anti_tackle
        from titanium import instant_pass

        _c_danger = And(
            team_has,
            positioning.opponent_in_pass_danger(carrier, opponents, r_int, carrier_stam),
        )
        at_walk, at_dbg = anti_tackle.carrier_walk_target(
            carrier,
            opp_goal,
            ball,
            opponents,
            r_int,
            team_goal,
            carrier_stam,
            _c_danger,
        )
        carrier_urgent = And(
            team_has,
            Or(at_dbg["need_pass"], Or(at_dbg["own_goal_push"], at_dbg["retreating"])),
        )
        weak_z = support_outlets.lower_density_flank_z(opponents)
        left_o, right_o, trail_o = support_outlets.at_safe_flank_stations(
            carrier,
            ball,
            opp_goal,
            opp_left_post,
            opp_right_post,
            team_goal,
            opponents,
            r_int,
            r_eff,
            carrier_stam,
            danger=_c_danger,
            at_debug=at_dbg,
            weak_z=weak_z,
        )
        raw_support = [left_o, right_o, trail_o]
        debug_viz.plot_xz("Titanium.Support.Left", "Yellow", left_o)
        debug_viz.plot_xz("Titanium.Support.Right", "Yellow", right_o)
        debug_viz.plot_xz("Titanium.Support.Trail", "Yellow", trail_o)

        mates_all = outfield + [p4]
        mate_stams_all = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 5)]
        carry_shared, release_shared, carry_meta = build_carrier_move(
            1,
            carrier,
            ball,
            opp_goal,
            team_goal,
            opp_left_post,
            opp_right_post,
            opponents,
            mates_all,
            mate_stams_all,
            r_int,
            r_eff,
            team_has,
            carrier_charge,
            attack_walk_in_open,
            attack_walk_in_pt,
            at_walk=at_walk,
            at_debug=at_dbg,
        )
        allow_teammate_steal = And(
            carrier_urgent,
            And(Not(carry_meta["kick_escape"]), Not(attack_walk_in_open)),
        )
        _recv_ok, _recv_dir, steal_target = instant_pass.best_instant_handoff(
            carrier,
            mates_all,
            mate_stams_all,
            opponents,
            r_int,
            carrier_stam,
            opp_goal,
            opp_left_post,
            opp_right_post,
            r_eff,
            require_shot=Bool(False),
        )
        _ = (_recv_ok, _recv_dir)
    else:
        carrier_urgent = Bool(False)
        allow_teammate_steal = Bool(False)
        carry_shared = None
        release_shared = None
        steal_target = carrier
        raw_support = []
        for ahead, side in ((9.0, 7.0), (9.0, -7.0), (-6.0, 0.0)):
            raw_support.append(ball + fwd * Float(ahead) + lat * Float(side))

    # Opponent can score by walking into our mouth — shot cover cannot stop it;
    # only a tackle can. Escalate every outfielder onto the ball when open.
    walk_in_open, walk_pt = unopposed_walk_in(ball, left_post, right_post, team4)
    walk_in_threat = And(opp_has, walk_in_open)
    # Loose ball on a path into OUR net (same model as GK sprint gate).
    goal_half_w = Abs(left_post.z)
    shot_threat_raw, shot_threat_t, shot_threat_pt = own_goal_threat(
        ball, ball_vel, team_goal.x, goal_half_w
    )
    net_shot_threat = And(loose, shot_threat_raw)
    # Time until a walk-in carrier reaches our mouth (carrier walks).
    walk_in_score_t = Distance(ball, walk_pt) / Float(CARRIER_SPEED)

    # One saver for a net-bound ball: fastest walker who makes it, else
    # fastest sprinter; sprint is last resort and only for that saver.
    net_saver_flags, net_sprint_flags = assign_fastest_net_saver(
        all_bodies, shot_threat_pt, shot_threat_t
    )

    # Tackle assignment once for the whole team (not once per outfielder).
    duty_flags, chaser_flags, press_body = tackle_plan()

    # Soft space-claim values (who keeps a contested station).
    # Emergency only for the assigned net saver / walk-in presser.
    move_pris = []
    for i in range(4):
        is_press = CompareFloats(Distance(all_bodies[i], press_body), Float(0.4), "<")
        emergency = Or(
            And(walk_in_threat, Or(duty_flags[i], is_press)),
            And(net_shot_threat, net_saver_flags[i]),
        )
        move_pris.append(
            positioning.task_value(
                has_flags[i],
                team_has,
                opp_has,
                loose,
                closest_flags[i],
                duty_flags[i],
                is_press,
                emergency=emergency,
            )
        )

    for slot, me, marks, raw in (
        (1, p1, opponents[0], raw_support[0]),
        (2, p2, opponents[1], raw_support[1]),
        (3, p3, opponents[2], raw_support[2]),
    ):
        has = SoccerGetBool(f"Team Player {slot} Has Ball")
        closest = SoccerGetBool(f"Is Team Player {slot} Closest Teammate to Ball")

        if WITH_ANTI_TACKLE:
            carry, shoot_now = carry_shared, release_shared
        else:
            mates = [p for i, p in enumerate(outfield) if i != (slot - 1)]
            mate_stams = [
                SoccerGetFloat(f"Team Player {i} Stamina")
                for i in range(1, 4)
                if i != slot
            ]
            mates_all_slot = mates + [p4]
            mate_stams_all_slot = mate_stams + [SoccerGetFloat("Team Player 4 Stamina")]
            charge = SoccerGetFloat(f"Teammate {slot} Shot Charge")
            carry, shoot_now, _meta = build_carrier_move(
                slot,
                me,
                ball,
                opp_goal,
                team_goal,
                opp_left_post,
                opp_right_post,
                opponents,
                mates_all_slot,
                mate_stams_all_slot,
                r_int,
                r_eff,
                has,
                charge,
                attack_walk_in_open,
                attack_walk_in_pt,
            )
        # Defensive default: seal our assigned opponent's shot cone.
        move = threat_cover(marks, team_goal, left_post, right_post, r_int)
        # Opponent carrier: interceptors chase. Higher-value press keeps route;
        # cover / other chasers shift if they claim the same space.
        duty = duty_flags[slot - 1]
        tackle_chase = chaser_flags[slot - 1]
        my_pri = move_pris[slot - 1]
        peer_bodies = [all_bodies[i] for i in range(4) if i != (slot - 1)]
        peer_pris = [move_pris[i] for i in range(4) if i != (slot - 1)]
        # Cover also soft-resolves vs press so shot-angle doesn't sit on the
        # intercept route (utility, not collision).
        cover_resolved = positioning.resolve_space_claims(
            me, move, peer_bodies, peer_pris, my_pri, r_int
        )
        move = ConditionalSetVector3(opp_has, cover_resolved, move)
        cutoff = held_ball_cutoff(me, ball, ball_vel, r_int)
        yield_chase = positioning.resolve_space_claims(
            me, cutoff, peer_bodies, peer_pris, my_pri, r_int
        )
        chase_move = ConditionalSetVector3(duty, cutoff, yield_chase)
        move = ConditionalSetVector3(And(opp_has, tackle_chase), chase_move, move)
        # Attacking outlets: mild equal-role spread; collapse toward carrier
        # when AT demands an emergency handoff.
        support = positioning.resolve_space_claims(
            me,
            raw,
            peer_bodies,
            peer_pris,
            my_pri,
            r_int,
            ball=ball,
            opp_goal=opp_goal,
            carrier_retreating=carrier_urgent,
            collapse_ok=carrier_urgent,
        )
        move = ConditionalSetVector3(And(team_has, Not(has)), support, move)
        move = ConditionalSetVector3(has, carry, move)

        # Teammate steal only when AT pushback and kick-pass cannot escape.
        i_am_steal = CompareFloats(Distance(me, steal_target), Float(0.35), "<")
        instant_steal = And(
            And(And(team_has, Not(has)), allow_teammate_steal),
            i_am_steal,
        )

        # Loose / flying ball.
        # Net-bound: only the single fastest saver claims it; sprint only if
        # that saver cannot walk there before the crossing. Else closest walks.
        meet_point_walk = predict_ball_meet_point(me, ball, ball_vel, WALK_SPEED)
        meet_point_sprint = predict_ball_meet_point(me, ball, ball_vel, SPRINT_SPEED)
        i_am_net_saver = net_saver_flags[slot - 1]
        net_needs_sprint = net_sprint_flags[slot - 1]
        sprint_save_shot = And(net_shot_threat, And(i_am_net_saver, net_needs_sprint))
        d_ball = Distance(me, ball)
        sprint_save_walkin = And(
            walk_in_threat,
            And(
                CompareFloats(d_ball / Float(WALK_SPEED), walk_in_score_t, ">"),
                CompareFloats(d_ball / Float(SPRINT_SPEED), walk_in_score_t, "<="),
            ),
        )
        sprint = Or(sprint_save_shot, sprint_save_walkin)
        loose_target = ConditionalSetVector3(
            sprint_save_shot, meet_point_sprint, meet_point_walk
        )
        claim_loose = Or(closest, And(net_shot_threat, i_am_net_saver))
        move = ConditionalSetVector3(And(loose, claim_loose), loose_target, move)
        debug_viz.plot_xz(f"Titanium.P{slot}.Pos", "Cyan", me)
        debug_viz.plot_xz(f"Titanium.P{slot}.Target", "Yellow", move)
        debug_viz.draw_player_move(me, move)
        SoccerController(
            slot,
            move,
            sprint,
            player_interact(slot, has, shoot_now, move, sprint, instant_steal),
        )

    # --- Player 4 goalkeeper ---
    h4 = SoccerGetBool("Team Player 4 Has Ball")
    c4 = SoccerGetFloat("Teammate 4 Shot Charge")
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
        claim_net_threat=And(net_shot_threat, net_saver_flags[3]),
    )
    debug_viz.draw_gk_branches(gk_debug)
    # gk_policy already owns loose claim + net-bound panic (meet-point chase).
    # Do not override with live ball position — that turns saves into pure pursuit.
    debug_viz.plot_xz("Titanium.P4.Pos", "Cyan", p4)
    debug_viz.plot_xz("Titanium.P4.Target", "Yellow", move4)
    debug_viz.draw_player_move(p4, move4)
    SoccerController(4, move4, sprint4, interact4)
