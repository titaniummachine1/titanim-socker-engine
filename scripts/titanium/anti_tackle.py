"""Held-ball rotation dodge — runs ONCE per tick.

Probes headings; marks which end-of-tick ball positions would be tackled
(stam-capable opps only). Among SAFE probes, pick the heading that
maximizes walk eval (progress + corridor + clearance). If no safe heading,
caller should pass to a teammate.
"""
from __future__ import annotations

import math

from titanium._env import *  # noqa: F401,F403
from titanium.constants import FIXED_DT, HOLD_OFFSET, KEEP_BALL_DANGER_PENALTY, WALK_SPEED
from titanium.geometry import rotate_xz, unit_or_zero
from titanium import positioning

_PROBE_OFFSETS = (
    Float(0),
    Float(-math.pi / 2),
    Float(math.pi / 2),
    Float(-math.pi / 4),
    Float(math.pi / 4),
)
# Safety margin added to EACH side of a blocked arc, in radians. One degree:
# enough to swallow float noise on the boundary without meaningfully narrowing
# the angles actually available.
ARC_MARGIN = math.radians(1.0)
_UNSAFE_EVAL = Float(-1000)
_SCORE_EVAL = Float(1e6)
_WALK = Float(WALK_SPEED)
_DT = Float(FIXED_DT)

step_toward = positioning.step_toward
simulate_end_of_tick = positioning.simulate_end_of_tick


def _opp_stamina(slot: int):
    return SoccerGetFloat(f"Opponent Player {slot} Stamina")


def opponent_staminas():
    return [_opp_stamina(i) for i in range(1, 5)]


def _can_tackle_us(opp_stam, my_stam):
    """Tackle duel: only stam >= carrier can dispossess. Lower stam = ghost for AT."""
    return CompareFloats(opp_stam, my_stam, ">=")


def _opponent_near_carrier(opp, ball_start, me, r_int):
    two_r = r_int * Float(2)
    near_me = CompareFloats(Distance(me, opp), two_r, "<=")
    near_ball = CompareFloats(Distance(ball_start, opp), two_r, "<=")
    return Or(near_me, near_ball)


def _end_tick_tackleable(ball_end, ball_start, opponents, staminas, r_int, me, my_stam):
    """Low-stam opps are non-threats — walk straight through them (AT ignores)."""
    threat = Bool(False)
    worst_stam = Float(-1)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        near_zone = _opponent_near_carrier(opp, ball_start, me, r_int)
        # Ghosts: not capable → never threaten, never affect AT.
        tracks = And(capable, near_zone)
        opp_end = step_toward(opp, ball_start, _WALK, _DT)
        near = CompareFloats(Distance(opp_end, ball_end), r_int, "<=")
        threatens = And(tracks, near)
        threat = Or(threat, threatens)
        worst_stam = ConditionalSetFloat(
            And(threatens, CompareFloats(stam, worst_stam, ">")),
            stam,
            worst_stam,
        )
    return threat, worst_stam


def _min_stam_capable_clearance(ball_end, ball_start, opponents, staminas, me, my_stam, r_int):
    """Clearance only vs stam-capable threats — ghosts do not steer the walk."""
    best = Float(1e6)
    for opp, stam in zip(opponents, staminas):
        capable = _can_tackle_us(stam, my_stam)
        near_zone = _opponent_near_carrier(opp, ball_start, me, r_int)
        tracks = And(capable, near_zone)
        opp_end = step_toward(opp, ball_start, _WALK, _DT)
        d = Distance(opp_end, ball_end)
        d_eff = ConditionalSetFloat(tracks, d, Float(1e6))
        best = ConditionalSetFloat(CompareFloats(d_eff, best, "<"), d_eff, best)
    return best


def probe_walk_eval(me, me_end, ball_end, heading, opp_goal, clearance, danger):
    """Maximize: progress to goal + face-goal geometry + clearance − danger keep penalty."""
    to_goal = unit_or_zero(opp_goal - me)
    progress = DotProduct(heading, to_goal)  # [-1,1] toward goal
    closer = Distance(me, opp_goal) - Distance(me_end, opp_goal)
    face = DotProduct(unit_or_zero(ball_end - me_end), unit_or_zero(opp_goal - me_end))
    margin = clearance
    keep_pen = ConditionalSetFloat(danger, Float(KEEP_BALL_DANGER_PENALTY), Float(0))
    return progress * Float(2) + closer * Float(0.5) + face + margin * Float(0.25) - keep_pen


@cache
def hold_block_arc(me, opp, r_int, my_stam, opp_stam, hold=HOLD_OFFSET):
    """The arc of aim directions this ONE opponent takes the ball on.

    Turning to aim parks the held ball at `me_end + heading*hold`, so as the aim
    sweeps, the ball traces a circle of radius `hold` around the carrier. The
    directions that land it within `r_int` of the opponent are therefore an ARC,
    centred on the direction to them, of half-width

        acos( (hold^2 + d^2 - r_int^2) / (2*hold*d) )

    That is the exact min/max angle a binary search converges toward, in one
    `acos` instead of five probe evaluations — and it is a true bound rather
    than the nearest of five samples.

    Two cheap outcomes fall out of the same triangle and are what make this
    worth doing at all:
      * `d > hold + r_int`  -> the circle never reaches them, arc is EMPTY.
        This is the bail: a distant opponent costs one compare, no trig.
      * `d < r_int - hold`  -> the circle lies entirely inside their reach, so
        EVERY direction is blocked.

    Stamina gates it: an opponent who cannot win the duel (`opp_stam < my_stam`)
    is a ghost and blocks nothing, matching `_can_tackle_us`.

    Returns `(toward, cos_half, blocks_everything, blocks_nothing)`, shaped like
    `geometry.opponent_invariants` so the per-candidate test stays a dot product.
    """
    opp_end = step_toward(opp, me, _WALK, _DT)
    offset = opp_end - me
    d = Magnitude(offset)
    toward = unit_or_zero(offset)
    h = Float(hold)

    capable = _can_tackle_us(opp_stam, my_stam)
    # Out of reach entirely — the ball circle never intersects their bubble.
    out_of_range = CompareFloats(d, h + r_int, ">")
    blocks_nothing = Or(Not(capable), out_of_range)
    # Swallowed — every heading puts the ball inside their reach.
    blocks_everything = And(capable, CompareFloats(d + h, r_int, "<="))

    denom = MultiplyFloats(Float(2.0) * h, d)
    safe_denom = ConditionalSetFloat(CompareFloats(denom, Float(1e-4), "<"), Float(1e-4), denom)
    # Outside [-1,1] means the circles do not intersect; the two flags above
    # already cover those cases, so the clamp only keeps the comparison sane.
    # Native ClampFloat — one node instead of four.
    cos_half = ClampFloat(
        (h * h + d * d - MultiplyFloats(r_int, r_int)) / safe_denom,
        Float(0) - Float(1),
        Float(1),
    )
    # Widen the blocked arc by ARC_MARGIN on EACH side, so an aim that only just
    # clears a defender is treated as blocked. A candidate landing exactly on the
    # boundary is otherwise decided by float error, and the cost of being wrong
    # is asymmetric: aim slightly too tight and the ball is gone, aim slightly
    # too wide and you give up a sliver of angle.
    #
    # Done with the cosine addition identity rather than acos/cos:
    #     cos(a + m) = cos a * cos m - sin a * sin m,  sin a = sqrt(1 - cos^2 a)
    # `cos m` and `sin m` are compile-time constants, so widening costs four
    # nodes and still never evaluates a trig function at runtime. A wider arc is
    # a SMALLER cosine, which is why this subtracts.
    sin_half = Sqrt(ClampFloat(Float(1) - MultiplyFloats(cos_half, cos_half), Float(0), Float(1)))
    cos_half = ClampFloat(
        MultiplyFloats(cos_half, Float(math.cos(ARC_MARGIN)))
        - MultiplyFloats(sin_half, Float(math.sin(ARC_MARGIN))),
        Float(0) - Float(1),
        Float(1),
    )
    return toward, cos_half, blocks_everything, blocks_nothing


@cache
def hold_direction_blocked(direction, arc):
    """True if aiming `direction` puts the held ball in this opponent's reach.

    `arc` is one `hold_block_arc` result. Inside the arc means the angle to
    `toward` is smaller than the half-width, i.e. its cosine is LARGER than
    `cos_half` — the same dot-product form `geometry.direction_forbidden` uses.
    """
    toward, cos_half, blocks_everything, blocks_nothing = arc
    inside = CompareFloats(DotProduct(direction, toward), cos_half, ">=")
    return And(Not(blocks_nothing), Or(blocks_everything, inside))


@cache
def arc_edges(arc):
    """The two boundary directions of a blocked arc — its exact min and max
    tackleable angle — WITHOUT ever computing the angle.

    `hold_block_arc` already produced `cos_half`, and `sin_half` is
    `sqrt(1 - cos^2)`, so rotating `toward` by +-half is just the rotation
    matrix with those two values substituted in. No `acos`, no `cos`, no `sin`
    at runtime:

        edge+ = ( x*cos - z*sin,  0,  x*sin + z*cos )
        edge- = ( x*cos + z*sin,  0, -x*sin + z*cos )

    These are the candidate headings worth trying when the straight line to goal
    is blocked: the closest you can aim to your intended direction while still
    grazing past this defender.
    """
    toward, cos_half, _blocks_everything, _blocks_nothing = arc
    parts = Vector3Split(toward)
    x, z = parts.x, parts.z
    sin_half = Sqrt(ClampFloat(Float(1) - MultiplyFloats(cos_half, cos_half), Float(0), Float(1)))
    xc, xs = MultiplyFloats(x, cos_half), MultiplyFloats(x, sin_half)
    zc, zs = MultiplyFloats(z, cos_half), MultiplyFloats(z, sin_half)
    left = Vector3(xc - zs, Float(0), xs + zc)
    right = Vector3(xc + zs, Float(0), zc - xs)
    return unit_or_zero(left), unit_or_zero(right)


def blocked_arcs(me, opponents, r_int, my_stam, hold=HOLD_OFFSET):
    """Every blocked arc, hoisted once for all candidate directions.

    TWO arcs per opponent, not one. The engine's tackle test is

        min(tackler_hold_to_ball, tackler_body_to_ball) <= interact_radius

    so a tackler reaches from their HOLD POINT as well as their body, and the
    hold point sits `hold` (1.67 m) ahead of them along their facing. Their real
    reach is therefore up to interact_radius + hold in the direction they face -
    nearly DOUBLE the 1.75 m body radius this model used to assume.

    That single omission is the defect, measured rather than guessed: against
    Poponeta over 180 s, 19 balls were lost to tackles, ZERO of them
    unavoidable, and ~90% were headings this model had judged safe. It was not
    unlucky, it was reading the wrong reach.

    A tackler's facing is not exposed by any getter, but the engine parks a held
    ball at `body + facing * hold`, so for a carrier the offset direction IS the
    facing. A defender without the ball has no such tell, so their facing is
    approximated as pointing at us - the direction that matters, since a
    defender facing away cannot reach us with their hold point anyway.
    """
    staminas = opponent_staminas()
    arcs = []
    for opp, stam in zip(opponents, staminas):
        # Body reach.
        arcs.append(hold_block_arc(me, opp, r_int, my_stam, stam, hold))
        # Hold-point reach: the same solve, from a point `hold` nearer to us.
        toward_us = unit_or_zero(me - opp)
        opp_hold = opp + toward_us * Float(hold)
        arcs.append(hold_block_arc(me, opp_hold, r_int, my_stam, stam, hold))
    return arcs


def best_open_direction(desired_dir, arcs):
    """Direction closest to `desired_dir` that no arc blocks.

    Walk as straight at the goal as the defence allows: try the direct line
    first, and if a defender owns it, fall back to the nearest angle that just
    clears somebody — which is exactly an arc edge. Scoring is the dot product
    with the desired direction, so "best" means "least deviation from straight
    at goal", and the winner is the tightest legal line rather than the best of
    five fixed offsets.

    Returns `(direction, any_open)`. `any_open` false means every candidate is
    covered: there is nowhere to walk without being tackled, so the caller
    should pass instead of retreating.
    """
    candidates = [desired_dir]
    for arc in arcs:
        left, right = arc_edges(arc)
        candidates.extend((left, right))

    best = desired_dir
    best_score = _UNSAFE_EVAL
    any_open = Bool(False)
    for cand in candidates:
        blocked = Bool(False)
        for arc in arcs:
            blocked = Or(blocked, hold_direction_blocked(cand, arc))
        open_ = Not(blocked)
        score = DotProduct(cand, desired_dir)
        better = And(open_, Or(Not(any_open), CompareFloats(score, best_score, ">")))
        best = ConditionalSetVector3(better, cand, best)
        best_score = ConditionalSetFloat(better, score, best_score)
        any_open = Or(any_open, open_)
    return best, any_open


def aim_is_safe(me, opponents, r_int, my_stam, hold=HOLD_OFFSET):
    """Build a predicate: can the carrier aim `direction` and keep the ball?

    Returned as a closure so `titanium.shot.clear_shot` / `clear_pass` can apply
    it per candidate without importing anti-tackle. The per-opponent arcs are
    hoisted here and shared across every candidate, so an extra aim direction
    costs one dot product and one compare per opponent — not another solve.
    """
    staminas = opponent_staminas()
    arcs = [
        hold_block_arc(me, opp, r_int, my_stam, stam, hold)
        for opp, stam in zip(opponents, staminas)
    ]

    def safe(direction):
        blocked = Bool(False)
        for arc in arcs:
            blocked = Or(blocked, hold_direction_blocked(direction, arc))
        return Not(blocked)

    return safe


def heading_safe(
    me,
    heading,
    ball_start,
    opponents,
    staminas,
    r_int,
    team_goal,
    my_stam,
    opp_goal,
    hold=HOLD_OFFSET,
    move_step=5.5,
):
    """Can the ball be HELD along `heading` for one tick without being taken?

    Turning to face a direction parks the held ball at `me_end + heading*hold`,
    so this is the question "if I point that way, do I still have the ball at
    the end of the tick". Used two ways: by `_probe` to score walk headings, and
    by `titanium.carrier` to reject shot directions the carrier could not
    actually aim along — see `titanium.shot.clear_shot`'s `direction_ok`.

    Enemy net = the ball is already a goal and cannot be tackled, so it counts
    safe. Own net is never safe, so no probe is tempted to place it there.
    """
    _me_end, ball_end = simulate_end_of_tick(me, heading, move_step, hold)
    tackleable, _worst = _end_tick_tackleable(
        ball_end, ball_start, opponents, staminas, r_int, me, my_stam
    )
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    tackle_threat = And(tackleable, Not(enemy_net))
    return Or(enemy_net, And(Not(tackle_threat), own_goal_ok))


def _probe(
    me,
    desired_dir,
    offset,
    ball_start,
    opponents,
    staminas,
    r_int,
    hold,
    move_step,
    team_goal,
    my_stam,
    opp_goal,
    danger,
):
    heading = rotate_xz(desired_dir, offset)
    me_end, ball_end = simulate_end_of_tick(me, heading, move_step, hold)
    tackleable, worst_stam = _end_tick_tackleable(
        ball_end, ball_start, opponents, staminas, r_int, me, my_stam
    )
    safe = heading_safe(
        me, heading, ball_start, opponents, staminas, r_int, team_goal, my_stam,
        opp_goal, hold, move_step,
    )
    # Recomputed rather than returned from `heading_safe`: `ball_in_goal_net`
    # is a cached graph function, so this is the same node, not a second one.
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    # Forward = eval-up toward goal. Scoring into their net is always usable.
    progress = DotProduct(heading, unit_or_zero(opp_goal - me))
    forward = CompareFloats(progress, Float(0), ">=")
    usable = Or(enemy_net, And(safe, forward))
    clearance = _min_stam_capable_clearance(
        ball_end, ball_start, opponents, staminas, me, my_stam, r_int
    )
    raw_eval = probe_walk_eval(me, me_end, ball_end, heading, opp_goal, clearance, danger)
    # Prefer placing the ball in their net over every ordinary walk eval.
    raw_eval = ConditionalSetFloat(enemy_net, _SCORE_EVAL, raw_eval)
    eval_score = ConditionalSetFloat(usable, raw_eval, _UNSAFE_EVAL)
    return safe, usable, heading, Abs(offset), tackleable, worst_stam, eval_score, progress


def search_safe_direction(
    me,
    desired,
    ball_start,
    opponents,
    r_int,
    team_goal,
    my_stam,
    danger=None,
    hold=HOLD_OFFSET,
    move_step=5.5,
):
    """Among safe+forward probes, pick max eval.

    If the only safe options go backward (hurt eval), need_pass — do not retreat.
    """
    if danger is None:
        danger = Bool(False)
    desired_dir = unit_or_zero(desired - me)
    staminas = opponent_staminas()
    opp_goal = desired

    # Closed-form pick: walk as straight at goal as the defence allows.
    # Candidates are the direct line plus every arc edge, scored by how little
    # they deviate from it. This replaces choosing among five fixed offsets,
    # each of which cost a full end-of-tick simulation to evaluate; an arc edge
    # is the EXACT tightest line past a defender, not the nearest of 5 samples.
    arcs = blocked_arcs(me, opponents, r_int, my_stam, hold)
    open_dir, any_open = best_open_direction(desired_dir, arcs)

    # Headings offered to `support_outlets`. Deliberately the five fixed offsets
    # rather than every arc edge: the arcs decide where the CARRIER walks, but
    # supports only need a spread of directions to stand on, and one station is
    # built per heading — nine candidates cost more in stations than they buy in
    # coverage (measured: 5476 -> 5646 nodes for no behavioural gain).
    #
    # Only `safe`, `usable` and `heading` survive DCE here; the eval and
    # clearance machinery each probe used to compute is dead once the arcs pick
    # the direction, so the simulation cost goes with it.
    probes = [
        _probe(
            me,
            desired_dir,
            off,
            ball_start,
            opponents,
            staminas,
            r_int,
            hold,
            move_step,
            team_goal,
            my_stam,
            opp_goal,
            danger,
        )
        for off in _PROBE_OFFSETS
    ]

    # The arcs decide WHERE to walk. The probes are still built because
    # `support_outlets` reads their per-heading safety and `debug_viz` plots
    # them, but they no longer select the direction.
    any_safe = any_open
    any_usable = any_open
    best_dir = open_dir
    best_eval = DotProduct(open_dir, desired_dir)
    best_offset = Float(0)

    # Own-goal guard still needs the actual end-of-tick ball position: the arcs
    # answer "can I keep the ball", not "does keeping it put it in my own net".
    _me_end, ball_end = simulate_end_of_tick(me, best_dir, move_step, hold)
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    predicted_safe = Or(enemy_net, And(any_open, own_goal_ok))
    # No forward-safe walk (only back / none) → pass, never walk backward.
    # Exception: enemy-net finish is usable even if classified oddly.
    need_pass = Not(any_usable)

    return best_dir, {
        "any_safe": any_safe,
        "any_forward": any_usable,
        "predicted_safe": predicted_safe,
        "probes": probes,
        "chosen_offset": best_offset,
        "best_eval": best_eval,
        "need_pass": need_pass,
    }


def carrier_walk_target(
    me,
    desired,
    ball_start,
    opponents,
    r_int,
    team_goal,
    my_stam,
    danger=None,
    step=5.5,
    hold=HOLD_OFFSET,
):
    direction, debug = search_safe_direction(
        me, desired, ball_start, opponents, r_int, team_goal, my_stam, danger, hold, step
    )
    return me + direction * Float(step), debug
