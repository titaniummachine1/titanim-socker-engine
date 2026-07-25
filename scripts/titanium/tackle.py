"""Tackle duty and the Interact policy that drives it."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403


def tackle_duty(slot: int):
    """Should THIS player be the one to press Interact on an opponent carrier?

    A tackle drains `min(tackler_stam, carrier_stam)` from BOTH players and is
    won by the tackler iff `tackler_stam >= carrier_stam`. Two consequences,
    neither of which we were using while everyone simply spammed Interact:

      * Sending everybody in wastes our strongest players. Among those who
        would actually WIN the duel, the *cheapest* winner should go; the
        others keep their stamina, and with it their own immunity to being
        tackled (a full-stamina player cannot be dispossessed at all).
      * When nobody can win outright, a doomed tackle is still worth making:
        it costs the carrier exactly as much as it costs us. Feed the weakest
        player in first and the carrier is ground down until somebody who was
        previously below the bar now clears it.

    So: if anyone in range can win, the lowest-stamina winner tackles;
    otherwise the lowest-stamina player in range tackles as a sacrifice.

    Ties break by slot through a tiny index epsilon on the sort key, so
    exactly one player is ever designated — without it, two equal-stamina
    players would both see themselves as "the cheapest" and both pile in,
    which is the waste this is meant to prevent.
    """
    carrier_stam = SoccerGetFloat("Ball Carrier Stamina")
    near = [SoccerGetBool(f"Is Ball Nearby Team Player {i}") for i in range(1, 5)]
    stam = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 5)]
    key = [stam[i] + Float((i + 1) * 0.0001) for i in range(4)]
    win = [And(near[i], CompareFloats(stam[i], carrier_stam, ">=")) for i in range(4)]

    any_win = Bool(False)
    for w in win:
        any_win = Or(any_win, w)
    # Prefer winners when one exists; otherwise everyone in range is a
    # candidate sacrifice.
    eligible = [ConditionalSetBool(any_win, win[i], near[i]) for i in range(4)]

    me = slot - 1
    duty = eligible[me]
    for k in range(4):
        if k == me:
            continue
        cheaper = And(eligible[k], CompareFloats(key[k], key[me], "<"))
        duty = And(duty, Not(cheaper))
    return duty


def player_interact(player: int, has_ball: Node, shoot_now: Node):
    """Interact policy: hold maximum charge permanently, release only to shoot.

    The engine charges while Interact is true, clamps at 1.0 with no decay
    and no max-hold timer, and fires the kick the frame Interact goes false —
    along that frame's MoveTo, not facing. Crucially, charging costs nothing:
    stamina only drains while *sprinting*. So a carrier should sit on a fully
    charged shot indefinitely and spend it the instant a lane opens, instead
    of dumping the ball at a fixed charge threshold the way this used to
    (which fired at ~0.7 charge in whatever direction MoveTo happened to
    point, and fired again on reaching 1.0).

    Claiming splits by ball state. A LOOSE ball is free — no duel, no cost,
    so anyone in range grabs it. A ball HELD by an opponent costs a stamina
    duel, so only the designated tackler presses (see `tackle_duty`); the
    rest keep their stamina instead of throwing it away on duels they were
    always going to lose.
    """
    nearby = SoccerGetBool(f"Is Ball Nearby Team Player {player}")
    loose = SoccerGetBool("Is Ball Loose")
    opp_has = SoccerGetBool("Opponent Has Ball")
    hold_charge = And(has_ball, Not(shoot_now))
    grab = And(nearby, loose)
    tackle = And(And(nearby, opp_has), tackle_duty(player))
    return Or(hold_charge, Or(grab, tackle))
