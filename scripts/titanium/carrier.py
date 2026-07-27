"""Ball-carrier: shoot if free; else instant tackle-pass; else AT walk; else kick pass."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import FULL_CHARGE, KEEP_BALL_DANGER_PENALTY
from titanium.shot import best_escape_pass, clear_shot, walk_target


def build_carrier_move(
    slot,
    me,
    ball,
    opp_goal,
    team_goal,
    opp_left_post,
    opp_right_post,
    opponents,
    teammates,
    teammate_stams,
    r_int,
    r_eff,
    has_ball,
    charge,
    at_walk=None,
    at_debug=None,
):
    """AT walk; kick-pass to escape; teammate steal only if kick cannot.

    Pass shared `at_walk` / `at_debug` from one `carrier_walk_target` so the
    graph does not rebuild anti-tackle once per outfielder.
    """
    from titanium import anti_tackle, debug_viz, instant_pass, positioning
    from titanium._env import WITH_ANTI_TACKLE  # noqa: F401

    my_stam = SoccerGetFloat("Ball Carrier Stamina")

    aimable = anti_tackle.aim_is_safe(me, opponents, r_int, my_stam)

    lane_ok, aim = clear_shot(
        me, opp_goal, opp_left_post, opp_right_post, opponents, r_eff,
        direction_ok=aimable,
    )
    ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    shoot_now = And(has_ball, And(lane_ok, ready))

    danger = And(
        has_ball,
        positioning.opponent_in_pass_danger(me, opponents, r_int, my_stam),
    )
    mates = list(teammates)
    stams = list(teammate_stams)

    if WITH_ANTI_TACKLE:
        if at_walk is None or at_debug is None:
            walk, at_debug = anti_tackle.carrier_walk_target(
                me, opp_goal, ball, opponents, r_int, team_goal, my_stam, danger
            )
        else:
            walk = at_walk
        # AT walking us into our net or backward → emergency handoff (no shot req).
        urgent = Or(
            at_debug["need_pass"],
            Or(at_debug["own_goal_push"], at_debug["retreating"]),
        )
    else:
        walk = walk_target(me, opp_goal)
        at_debug = None
        urgent = danger

    can_pass, pass_dir, _pass_mate = best_escape_pass(
        me, mates, opponents, r_eff, direction_ok=aimable
    )
    # Kick can still get the ball out → never teammate-tackle.
    kick_escape = And(ready, can_pass)

    # Teammate steal ONLY when AT pushback cannot be stopped by a kick-pass.
    inst_any, inst_dir, _ = instant_pass.best_instant_handoff(
        me,
        mates,
        stams,
        opponents,
        r_int,
        my_stam,
        opp_goal,
        opp_left_post,
        opp_right_post,
        r_eff,
        require_shot=Bool(False),
    )
    instant_now = And(
        has_ball,
        And(urgent, And(inst_any, And(Not(shoot_now), Not(kick_escape)))),
    )

    if WITH_ANTI_TACKLE:
        want_pass = Or(urgent, danger)
        pass_now = And(
            has_ball,
            And(ready, And(can_pass, And(want_pass, And(Not(shoot_now), Not(instant_now))))),
        )
        at_debug = dict(at_debug)
        at_debug["pass_danger"] = danger
        at_debug["urgent_handoff"] = urgent
        at_debug["keep_penalty"] = ConditionalSetFloat(
            And(danger, Not(Or(shoot_now, Or(pass_now, instant_now)))),
            Float(KEEP_BALL_DANGER_PENALTY),
            Float(0),
        )
        debug_viz.plot_anti_tackle(slot, has_ball, at_debug)
    else:
        pass_now = And(
            has_ball,
            And(ready, And(can_pass, And(danger, And(Not(shoot_now), Not(instant_now))))),
        )

    release_now = Or(shoot_now, pass_now)
    shoot_to = me + aim * Float(10)
    pass_to = me + pass_dir * Float(10)
    inst_to = me + inst_dir * Float(10)
    move = ConditionalSetVector3(shoot_now, shoot_to, walk)
    move = ConditionalSetVector3(pass_now, pass_to, move)
    move = ConditionalSetVector3(instant_now, inst_to, move)
    chase = walk_target(me, ball, step=8.0)
    move = ConditionalSetVector3(has_ball, move, chase)
    return move, release_now, {
        "urgent": urgent,
        "kick_escape": kick_escape,
    }
