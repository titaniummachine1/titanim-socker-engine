"""Ball-carrier: shoot if free; else AT max-eval forward walk; else pass (never retreat)."""
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
    r_int,
    r_eff,
    has_ball,
    charge,
):
    """AT picks safe+forward max-eval walk. If only back is safe (or none) → pass."""
    from titanium import anti_tackle, debug_viz, positioning
    from titanium._env import WITH_ANTI_TACKLE

    lane_ok, aim = clear_shot(me, opp_goal, opp_left_post, opp_right_post, opponents, r_eff)
    ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    shoot_now = And(has_ball, And(lane_ok, ready))

    # Carrier stam for AT/ghost filter — Ball Carrier is authoritative while holding.
    my_stam = SoccerGetFloat("Ball Carrier Stamina")
    danger = And(
        has_ball,
        positioning.opponent_in_pass_danger(me, opponents, r_int, my_stam),
    )
    mates = list(teammates)
    can_pass, pass_dir, _pass_mate = best_escape_pass(me, mates, opponents, r_eff)

    if WITH_ANTI_TACKLE:
        walk, at_debug = anti_tackle.carrier_walk_target(
            me, opp_goal, ball, opponents, r_int, team_goal, my_stam, danger
        )
        # need_pass: no safe forward walk (would only retreat) OR pass-danger.
        want_pass = Or(at_debug["need_pass"], danger)
        pass_now = And(has_ball, And(ready, And(can_pass, And(want_pass, Not(shoot_now)))))
        at_debug = dict(at_debug)
        at_debug["pass_danger"] = danger
        at_debug["keep_penalty"] = ConditionalSetFloat(
            And(danger, Not(Or(shoot_now, pass_now))),
            Float(KEEP_BALL_DANGER_PENALTY),
            Float(0),
        )
        debug_viz.plot_anti_tackle(slot, has_ball, at_debug)
    else:
        walk = walk_target(me, opp_goal)
        pass_now = And(has_ball, And(ready, And(can_pass, And(danger, Not(shoot_now)))))

    release_now = Or(shoot_now, pass_now)
    shoot_to = me + aim * Float(10)
    pass_to = me + pass_dir * Float(10)
    move = ConditionalSetVector3(shoot_now, shoot_to, walk)
    move = ConditionalSetVector3(pass_now, pass_to, move)
    chase = walk_target(me, ball, step=8.0)
    move = ConditionalSetVector3(has_ball, move, chase)
    return move, release_now
