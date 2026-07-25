"""Ball-carrier movement: walk a safe lane, snap onto the shot the instant
a lane opens."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import FULL_CHARGE
from titanium.shot import clear_shot, safe_walk_target


def build_carrier_move(me, ball, opp_goal, opp_left_post, opp_right_post, opponents, r_eff, has_ball, charge):
    """With ball: walk a safe lane toward goal sitting on a full charge, and
    on the tick a scoring lane opens snap MoveTo onto the aim direction.

    The snap *is* the aim — the engine kicks along MoveTo — so the release
    and the snap have to land on the same tick. That is why `shoot_now` is
    returned rather than recomputed: it must drive Interact too.

    The `charge` gate is not optional. Releasing means dropping Interact, and
    Interact is also what *builds* the charge — so firing the moment a lane
    happens to be open (which, on pickup in space, is immediately) holds the
    ball at zero charge forever: the engine needs >0.05 to kick at all, so
    nothing is ever struck. Wait for the charge to actually be banked, then
    spend it.
    """
    # Shot origin is the CARRIER, not the ball's transform: the real engine
    # visually orbits the held ball at an offset in front of the player, but
    # the actual kick fires from the player's own center. Aiming the tangent
    # geometry from `ball` (the offset position) instead of `me` was solving
    # a slightly wrong triangle every time.
    lane_ok, aim = clear_shot(me, opp_goal, opp_left_post, opp_right_post, opponents, r_eff)
    ready = CompareFloats(charge, Float(FULL_CHARGE), ">=")
    shoot_now = And(has_ball, And(lane_ok, ready))
    walk = safe_walk_target(me, ball, opp_goal, opponents, r_eff)
    shoot_to = me + aim * Float(10)
    move = ConditionalSetVector3(shoot_now, shoot_to, walk)
    # Without ball: chase ball with same cone (avoid running through tacklers).
    chase = safe_walk_target(me, ball, ball, opponents, r_eff, step=8.0)
    move = ConditionalSetVector3(has_ball, move, chase)
    return move, shoot_now
