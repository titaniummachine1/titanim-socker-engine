"""Tackle duty and the Interact policy that drives it."""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium.constants import CARRIER_SPEED, FIXED_DT, SPRINT_SPEED, WALK_SPEED
from titanium.geometry import closest_point_on_goal_mouth, pos, unit_or_zero


def _xz_dist(a, b):
    """Planar distance — Interact is XZ-only (height is bounce, not a claim gate)."""
    dx = a.x - b.x
    dz = a.z - b.z
    return Sqrt(dx * dx + dz * dz)


def _step_toward(me, target, speed):
    """Where a player ENDS this tick moving at `speed` toward `target`.

    A tick resolves DIRECTION, then MOVEMENT, and only then INTERACTION — so
    whether Interact can succeed depends on where the player will be, not where
    it is standing while deciding.
    """
    to = target - me
    d = Magnitude(to)
    step = Float(speed) * Float(FIXED_DT)
    travel = ConditionalSetFloat(CompareFloats(d, step, "<"), d, step)
    return me + ScaleVector3(unit_or_zero(to), travel)


def can_reach_ball(player: int, move, speed):
    """Would pressing Interact after this tick's movement get us the ball?

    Must use the SAME speed the controller is sprinting/walking at. Predicting
    walk while the player sprints underestimates travel — we never press even
    though the body is inside interact radius and the ball flies through.

    `Is Ball Nearby` is a WIDER radius; do not use it to claim (burns the
    rising-edge impulse before real reach).
    """
    r_int = SoccerGetFloat("Player Interact Radius")
    me_end = _step_toward(pos(f"Team Player {player}"), move, speed)
    ball_end = pos("Ball") + ScaleVector3(SoccerGetVector3("Ball Velocity"), Float(FIXED_DT))
    return CompareFloats(_xz_dist(ball_end, me_end), r_int, "<=")


def _walk_in_score_time(ball):
    """Seconds until opp carrier walks into OUR net via shortest mouth path."""
    left = pos("Team Goal Left Post")
    right = pos("Team Goal Right Post")
    goal_pt = closest_point_on_goal_mouth(ball, left, right)
    return Distance(ball, goal_pt) / Float(CARRIER_SPEED)


def _intercept_time(me, ball, r_int):
    """Walk time until this body enters interact range of the ball."""
    gap = Distance(me, ball) - r_int
    gap = ConditionalSetFloat(CompareFloats(gap, Float(0), "<"), Float(0), gap)
    return gap / Float(WALK_SPEED)


def _cheapest_among(mask, key, me):
    """True if `me` is in `mask` and has the lowest key among masked players."""
    best = mask[me]
    for k in range(4):
        if k == me:
            continue
        cheaper = And(mask[k], CompareFloats(key[k], key[me], "<"))
        best = And(best, Not(cheaper))
    return best


def tackle_plan():
    """Shared interceptor / bully / steal assignment for all four bodies.

    Returns `(duty_flags, chaser_flags, press_body)` — index 0 = P1.
    Call once per think; per-slot code just indexes the flags.
    """
    ball = pos("Ball")
    team_goal = pos("Team Goal Center")
    r_int = SoccerGetFloat("Player Interact Radius")
    t_score = _walk_in_score_time(ball)
    carrier_stam = SoccerGetFloat("Ball Carrier Stamina")
    # Bully only if we still have ≥ 25% of the carrier's stam — weaker drains
    # are useless (and repeating them is worse).
    bully_floor = carrier_stam * Float(0.25)

    bodies = [pos(f"Team Player {i}") for i in range(1, 5)]
    # Outfield marks Opp 1..3; GK has no mark (dist 0 → stays preferred cover).
    marks = [pos(f"Opponent Player {i}") for i in range(1, 4)] + [team_goal]
    stam = [SoccerGetFloat(f"Team Player {i} Stamina") for i in range(1, 5)]
    mark_dist = [Distance(marks[i], team_goal) for i in range(4)]
    # Prefer low stam; same stam → prefer larger mark_dist (less important threat).
    key_asc = [
        stam[i] - mark_dist[i] * Float(0.001) + Float((i + 1) * 0.00001)
        for i in range(4)
    ]

    can_int = []
    for i in range(4):
        t_arr = _intercept_time(bodies[i], ball, r_int)
        can_int.append(CompareFloats(t_arr, t_score, "<="))

    any_int = Bool(False)
    any_not_max = Bool(False)
    for i in range(4):
        any_int = Or(any_int, can_int[i])
        below_max = CompareFloats(stam[i], Float(0.999), "<")
        any_not_max = Or(any_not_max, And(can_int[i], below_max))
    all_maxed = And(any_int, Not(any_not_max))

    can_win = [
        And(can_int[i], CompareFloats(stam[i], carrier_stam, ">=")) for i in range(4)
    ]
    can_sac = [
        And(
            can_int[i],
            And(
                CompareFloats(stam[i], carrier_stam, "<"),
                CompareFloats(stam[i], bully_floor, ">="),
            ),
        )
        for i in range(4)
    ]

    any_win = Bool(False)
    any_sac = Bool(False)
    for i in range(4):
        any_win = Or(any_win, can_win[i])
        any_sac = Or(any_sac, can_sac[i])

    # Bully every weaker interceptor (time budget already in can_int) before
    # any steal — unless the whole intercept pack is already maxed.
    sac_phase = And(any_sac, Not(all_maxed))

    duty_flags = []
    for i in range(4):
        sac_i = _cheapest_among(can_sac, key_asc, i)
        win_i = _cheapest_among(can_win, key_asc, i)
        desper_i = _cheapest_among(can_int, key_asc, i)
        steal_i = ConditionalSetBool(any_win, win_i, desper_i)
        normal_i = ConditionalSetBool(sac_phase, sac_i, steal_i)
        duty_flags.append(ConditionalSetBool(all_maxed, desper_i, normal_i))

    press_body = bodies[0]
    for i in range(4):
        press_body = ConditionalSetVector3(duty_flags[i], bodies[i], press_body)
    return duty_flags, can_int, press_body


def tackle_roles(slot: int):
    """Who presses Interact vs who closes on the opponent carrier.

    Budget: shortest walk-in time to OUR goal mouth. Only teammates who can
    walk into interact range before that deadline are assigned.

    All interceptors maxed (≈1.0): bullying is pointless — tackle immediately
    with the least-stam interceptor, breaking ties toward whoever is guarding
    the least important mark (farthest from our goal; GK last).

    Otherwise, while any interceptor still has stam in [25% of carrier, carrier):
    they trade a bully within the budget (least-stam first). Below 25% of the
    carrier's stam a bully cannot meaningfully drain them — skip those bodies.
    Only after no eligible weaker interceptor remains do we steal — least-stam
    among those who can win (stam ≥ carrier). If nobody can win and bully is
    spent: least-stam interceptor still presses (no further staging).

    Returns `(duty, chaser, press_body)` — duty may Interact when in reach;
    chaser MoveTo ball while opponent holds; `press_body` is the teammate who
    currently presses (sacrifice / stealer) so others can yield spacing.
    """
    duty_flags, chaser_flags, press_body = tackle_plan()
    me = slot - 1
    return duty_flags[me], chaser_flags[me], press_body


def tackle_duty(slot: int):
    """Back-compat: press authorization only."""
    duty, _chaser, _press = tackle_roles(slot)
    return duty


def player_interact(
    player: int,
    has_ball: Node,
    shoot_now: Node,
    move=None,
    sprint=None,
    instant_steal=None,
):
    """Interact: charge while holding; pulse claim; or instant teammate steal.

    Engine claim/tackle is a rising edge on a per-player 64-bit Interact
    history (held now, not held last tick). Holding true only fires once.

    Do NOT use GetVariable/SetVariable to toggle — the sim settles SetVariables
    8× per think and a self-ref flip collapses to a stuck press. Drive the
    oscillation from Current Simulation Time instead (stable across settle).

    When our team has the ball, stay quiet except an emergency AT pushback
    handoff steal (`instant_steal`) — never voluntary teammate tackles.
    """
    _ = (move, sprint)
    hold_charge = And(has_ball, Not(shoot_now))
    team_has = SoccerGetBool("Team Has Ball")
    want_spam = And(Not(has_ball), Not(team_has))

    t = SoccerGetFloat("Current Simulation Time")
    dt = SoccerGetFloat("Fixed Delta Time")
    dt_safe = ConditionalSetFloat(CompareFloats(dt, Float(1e-6), "<"), Float(1e-6), dt)
    tick = Floor(t / dt_safe)
    half = Floor(tick * Float(0.5))
    odd_tick = CompareFloats(tick - half * Float(2), Float(0.5), ">")
    press = And(want_spam, odd_tick)
    steal = Bool(False) if instant_steal is None else And(instant_steal, odd_tick)
    return Or(hold_charge, Or(press, steal))
