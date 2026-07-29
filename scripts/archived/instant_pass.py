"""Emergency tackle-pass (Titanium only): last-resort handoff under AT pushback.

Default: never tackle a teammate. Only when anti-tackle has nowhere forward
and a kick-pass cannot escape, the carrier faces a mate so the hold offset
puts the ball in their interact radius; the mate presses Interact and steals
(equal/higher stam). The turn must be AT-safe so the ball is not intercepted
on the rotate.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import HOLD_OFFSET
from titanium.geometry import unit_or_zero
from titanium.shot import clear_shot


def handoff_to_mate(
    carrier,
    mate,
    mate_stam,
    opponents,
    r_int,
    carrier_stam,
    opp_goal,
    opp_left_post,
    opp_right_post,
    r_eff,
    require_shot=None,
):
    """Can we hand to `mate` this tick?

    Callers should only invoke this under AT pushback with no kick escape.
    `require_shot` defaults True (legacy); emergency path passes False.

    Returns `(ok, toward_mate)`.
    """
    from titanium import anti_tackle

    if require_shot is None:
        require_shot = Bool(True)

    toward = unit_or_zero(mate - carrier)
    aimable = anti_tackle.aim_is_safe(carrier, opponents, r_int, carrier_stam)
    safe_turn = aimable(toward)
    ball_aimed = carrier + toward * Float(HOLD_OFFSET)
    in_reach = CompareFloats(Distance(mate, ball_aimed), r_int, "<=")
    can_win = CompareFloats(mate_stam, carrier_stam, ">=")
    # Never treat the carrier body as its own steal target.
    not_self = CompareFloats(Distance(carrier, mate), Float(0.5), ">")
    base = And(safe_turn, And(in_reach, And(can_win, not_self)))
    recv_aim = anti_tackle.aim_is_safe(mate, opponents, r_int, mate_stam)
    lane_ok, _aim = clear_shot(
        mate,
        opp_goal,
        opp_left_post,
        opp_right_post,
        opponents,
        r_eff,
        direction_ok=recv_aim,
    )
    ok = ConditionalSetBool(require_shot, And(base, lane_ok), base)
    return ok, toward


def best_instant_handoff(
    carrier,
    mates,
    mate_stams,
    opponents,
    r_int,
    carrier_stam,
    opp_goal,
    opp_left_post,
    opp_right_post,
    r_eff,
    require_shot=None,
):
    """Nearest mate who can receive an AT-safe tackle-pass.

    Returns `(any_ok, toward, mate_pos)`.
    """
    if require_shot is None:
        require_shot = Bool(True)
    best_mate = mates[0]
    best_dir = unit_or_zero(mates[0] - carrier)
    best_d = Float(1e9)
    any_ok = Bool(False)
    for mate, stam in zip(mates, mate_stams):
        ok, toward = handoff_to_mate(
            carrier,
            mate,
            stam,
            opponents,
            r_int,
            carrier_stam,
            opp_goal,
            opp_left_post,
            opp_right_post,
            r_eff,
            require_shot=require_shot,
        )
        d = Distance(carrier, mate)
        better = And(ok, Or(Not(any_ok), CompareFloats(d, best_d, "<")))
        best_mate = ConditionalSetVector3(better, mate, best_mate)
        best_dir = ConditionalSetVector3(better, toward, best_dir)
        best_d = ConditionalSetFloat(better, d, best_d)
        any_ok = Or(any_ok, ok)
    return any_ok, best_dir, best_mate


def i_am_handoff_receiver(
    me,
    my_stam,
    carrier,
    mates,
    mate_stams,
    opponents,
    r_int,
    carrier_stam,
    opp_goal,
    opp_left_post,
    opp_right_post,
    r_eff,
    require_shot=None,
):
    """True if this player is the chosen instant-pass receiver this tick."""
    if require_shot is None:
        require_shot = Bool(True)
    ok_me, _ = handoff_to_mate(
        carrier,
        me,
        my_stam,
        opponents,
        r_int,
        carrier_stam,
        opp_goal,
        opp_left_post,
        opp_right_post,
        r_eff,
        require_shot=require_shot,
    )
    beaten = Bool(False)
    my_d = Distance(carrier, me)
    for mate, stam in zip(mates, mate_stams):
        ok_o, _ = handoff_to_mate(
            carrier,
            mate,
            stam,
            opponents,
            r_int,
            carrier_stam,
            opp_goal,
            opp_left_post,
            opp_right_post,
            r_eff,
            require_shot=require_shot,
        )
        closer = And(ok_o, CompareFloats(Distance(carrier, mate), my_d, "<"))
        beaten = Or(beaten, closer)
    return And(ok_me, Not(beaten))
