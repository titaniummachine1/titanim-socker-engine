"""Goalkeeper: safe cover stand, charge-aware press, threat interception,
and the multi-body defense that hands the same cover construction to
outfielders (see `threat_cover`, used from `titanium.graph`)."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.ball_physics import own_goal_threat, predict_ball_meet_point
from titanium.constants import (
    CARRIER_SPEED,
    FULL_CHARGE,
    POST_AIM_RADIUS,
    SHOT_CHARGE_TIME_S,
    SPRINT_SPEED,
    WALK_SPEED,
)
from titanium.geometry import (
    clamp01,
    clamp_own_half,
    is_legal_direction,
    opponent_half_angle,
    opponent_invariants,
    post_tangent,
    pos,
    unit_or_zero,
)
from titanium.shot import walk_target
from titanium.tackle import player_interact


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
    to_l, _ = post_tangent(shot_origin, left_post, POST_AIM_RADIUS)
    to_r, _ = post_tangent(shot_origin, right_post, POST_AIM_RADIUS)
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

    # Applied ONCE: `f(x) = seals(x) ? x : home` is idempotent. If seals(x)
    # then f(x)=x, so f(f(x))=f(x); otherwise f(x)=home and f(home) is home
    # either way. A second application was pure cost (126 nodes / 228 edges).
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
    claim_net_threat=None,
):
    """GK stays inside the shot-intercept safe region.

    Sprint only for a loose goal-bound ball that walk cannot reach in time,
    and only when this GK is the assigned net saver (`claim_net_threat`).

    Returns `(move, sprint, interact, debug)`.
    """
    if claim_net_threat is None:
        claim_net_threat = Bool(True)

    interact_r = SoccerGetFloat("Player Interact Radius")
    cover, seals = gk_cover_stand(ball, team_goal, left_post, right_post, interact_r)

    # With ball: clear upfield; walk straight toward the clear direction.
    clear_dir = SoccerGetVector3("Clear direction from Teammate 4")
    clear_target = gk + unit_or_zero(clear_dir) * Float(14)
    clear_target = ConditionalSetVector3(
        IsNull(clear_dir), pos("Opponent Goal Center"), clear_target
    )
    clear_spot = walk_target(gk, clear_target, step=14)
    move = ConditionalSetVector3(has_ball, clear_spot, cover)

    # Net-bound loose ball: only the assigned fastest saver claims it.
    goal_half_width_est = Abs(left_post.z)
    is_threat_raw, threat_time, threat_point = own_goal_threat(
        ball, ball_vel, team_goal.x, goal_half_width_est
    )
    is_threat = And(And(is_threat_raw, loose), claim_net_threat)
    dist_gk_threat = Distance(gk, threat_point)
    walk_time_threat = dist_gk_threat / Float(WALK_SPEED)
    walk_cant_make_it = CompareFloats(walk_time_threat, threat_time, ">")
    sprint = And(is_threat, walk_cant_make_it)

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

    meet_point_gk = predict_ball_meet_point(gk, ball, ball_vel, WALK_SPEED)
    meet_point_gk_sprint = predict_ball_meet_point(gk, ball, ball_vel, SPRINT_SPEED)
    threat_chase = ConditionalSetVector3(sprint, meet_point_gk_sprint, meet_point_gk)
    move = ConditionalSetVector3(is_threat, threat_chase, move)

    debug = {"is_threat": is_threat, "threat_point": threat_point, "opp_has": opp_has, "press": press, "move": move}

    nearby = SoccerGetBool("Is Ball Nearby Team Player 4")
    closest_gk = SoccerGetBool("Is Team Player 4 Closest Teammate to Ball")
    chase_target_gk = ConditionalSetVector3(nearby, ball, meet_point_gk)
    go_to_ball = Or(And(loose, closest_gk), And(nearby, Not(has_ball)))
    chase_ok = Or(nearby, seals(gk))
    ordinary_chase = And(And(go_to_ball, chase_ok), Not(is_threat))
    move = ConditionalSetVector3(ordinary_chase, chase_target_gk, move)
    move = ConditionalSetVector3(And(And(nearby, Not(has_ball)), Not(is_threat)), ball, move)

    move = ConditionalSetVector3(has_ball, move, clamp_own_half(move, team_goal))

    clear_aim = unit_or_zero(
        ConditionalSetVector3(IsNull(clear_dir), pos("Opponent Goal Center") - gk, clear_dir)
    )
    clear_invariants = [
        opponent_invariants(o, gk, r_eff, opponent_half_angle(o, gk, r_eff))
        for o in opponents
    ]
    gk_ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    gk_shoot = And(has_ball, And(is_legal_direction(clear_aim, clear_invariants), gk_ready))
    interact = player_interact(4, has_ball, gk_shoot, move, sprint)
    return move, sprint, interact, debug
