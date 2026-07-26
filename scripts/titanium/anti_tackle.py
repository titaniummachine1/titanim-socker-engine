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

    best_dir = desired_dir
    best_eval = _UNSAFE_EVAL
    best_offset = Float(0)
    any_usable = Bool(False)
    any_safe = Bool(False)

    for off, (safe, usable, direction, abs_off, _tackleable, _worst, eval_score, _prog) in zip(
        _PROBE_OFFSETS, probes
    ):
        any_safe = Or(any_safe, safe)
        better = And(usable, Or(Not(any_usable), CompareFloats(eval_score, best_eval, ">")))
        best_dir = ConditionalSetVector3(better, direction, best_dir)
        best_eval = ConditionalSetFloat(better, eval_score, best_eval)
        best_offset = ConditionalSetFloat(better, off, best_offset)
        any_usable = Or(any_usable, usable)

    _me_end, ball_end = simulate_end_of_tick(me, best_dir, move_step, hold)
    predicted_tackleable, _ = _end_tick_tackleable(
        ball_end, ball_start, opponents, opponent_staminas(), r_int, me, my_stam
    )
    enemy_net = positioning.ball_in_goal_net(ball_end, opp_goal)
    own_net = positioning.ball_in_goal_net(ball_end, team_goal)
    own_goal_ok = And(
        positioning.ball_clear_of_own_goal_plane(ball_end, team_goal),
        Not(own_net),
    )
    predicted_safe = Or(
        enemy_net,
        And(Not(And(predicted_tackleable, Not(enemy_net))), own_goal_ok),
    )
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
