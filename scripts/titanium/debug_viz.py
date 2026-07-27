"""All DebugDraw/TimePlot visualization, pulled out of the gameplay logic.

Every function here takes already-computed positions/values as parameters
and only calls Debug* / TimePlot side-effect nodes — nothing here feeds
back into a movement decision. That's the actual module boundary: gameplay
modules compute, this module only draws what they computed.
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium._env import traj
from titanium.constants import POST_AIM_RADIUS, SPRINT_SPEED
from titanium.ball_physics import airborne_hang_time, trajectory_breakpoints
from titanium.geometry import post_tangent
from titanium.shot import unopposed_walk_in

OFF_PITCH = Vector3(Float(999), Float(0), Float(999))


def plot_xz(prefix: str, color: str, vec) -> None:
    """Log a position/direction to TimePlot (F1) as two channels, X and Z —
    Y is always ~0 on this pitch so it's not worth the extra channel. Kept
    deliberately minimal (no Y, no derived values) so this doesn't add
    enough TimePlot load to visibly lag the real game.
    """
    comps = Vector3Split(vec)
    TimePlot(String(f"{prefix}.X"), color, String(""), comps.x)
    TimePlot(String(f"{prefix}.Z"), color, String(""), comps.z)


def draw_shot_cones(ball, team4, opponents, opp_left_post, opp_right_post, left_post, right_post):
    """True (post-tangent corrected) min/max straight-line shot cone at
    both goals. Green = cone into the enemy goal (what we could score with
    right now); red = cone into our own goal (the danger cone the GK must
    seal).

    Origin must be the actual kicker's BODY, not the raw ball position:
    when the ball is held, "Ball" reports a visually-offset point next to
    the carrier, not the carrier itself (same class of bug the shot-origin
    fix in `titanium.carrier`/`titanium.goalkeeper` already applies for
    real decisions) — a cone drawn from that offset point wouldn't match
    the angle the graph itself actually computes for any real decision.
    Falls back to the raw ball position only when it's genuinely loose (no
    explicit holder bit set on either side).
    """
    team_has_bits = [SoccerGetBool(f"Team Player {i} Has Ball") for i in range(1, 5)]
    opp_has_bits = [SoccerGetBool(f"Opponent Player {i} Has Ball") for i in range(1, 5)]
    shot_origin = ball
    for player, has in zip(team4, team_has_bits):
        shot_origin = ConditionalSetVector3(has, player, shot_origin)
    for opp, has in zip(opponents, opp_has_bits):
        shot_origin = ConditionalSetVector3(has, opp, shot_origin)

    _, enemy_aim_l = post_tangent(shot_origin, opp_left_post, POST_AIM_RADIUS)
    _, enemy_aim_r = post_tangent(shot_origin, opp_right_post, POST_AIM_RADIUS)
    DebugDrawLine(shot_origin, enemy_aim_l, Float(0.06), "Green")
    DebugDrawLine(shot_origin, enemy_aim_r, Float(0.06), "Green")
    DebugDrawDisc(enemy_aim_l, Float(0.15), Float(0.05), "Green")
    DebugDrawDisc(enemy_aim_r, Float(0.15), Float(0.05), "Green")

    _, own_aim_l = post_tangent(shot_origin, left_post, POST_AIM_RADIUS)
    _, own_aim_r = post_tangent(shot_origin, right_post, POST_AIM_RADIUS)
    DebugDrawLine(shot_origin, own_aim_l, Float(0.06), "Red")
    DebugDrawLine(shot_origin, own_aim_r, Float(0.06), "Red")
    DebugDrawDisc(own_aim_l, Float(0.15), Float(0.05), "Red")
    DebugDrawDisc(own_aim_r, Float(0.15), Float(0.05), "Red")


def draw_gk_branches(debug: dict) -> None:
    """The GK's threat-vs-press decision, as it happened this tick — Yellow
    where `own_goal_threat` wants to go, Magenta where the opponent-press
    branch wants to go, Blue where the GK is actually being sent. Off-pitch
    sentinel when a branch isn't the one active, so only the driving branch
    is visible on any given tick. `debug` is `gk_policy`'s returned dict."""
    DebugDrawDisc(ConditionalSetVector3(debug["is_threat"], debug["threat_point"], OFF_PITCH), Float(0.3), Float(0.15), "Yellow")
    DebugDrawDisc(ConditionalSetVector3(debug["opp_has"], debug["press"], OFF_PITCH), Float(0.3), Float(0.15), "Magenta")
    DebugDrawDisc(debug["move"], Float(0.25), Float(0.2), "Blue")


def draw_player_move(me, move) -> None:
    """Ground truth: exactly where this player is actually being commanded
    to walk/sprint, this tick — not a guess about intent."""
    DebugDrawLine(me, move, Float(0.05), "Orange")
    DebugDrawDisc(move, Float(0.2), Float(0.15), "Orange")


def predict_ball_path(ball, ball_vel, loose):
    """ALWAYS-ON predicted ball path: airborne glide, then up to 3 grounded
    friction+bounce legs — the exact same physics `own_goal_threat` uses,
    computed raw here (not through `ball_physics.call_event_leg_chain`'s
    CustomFunction, so there's no shared-variable-slot concern chaining 3
    legs inline).

    KNOWN LIMITATION, not fixed here: there is no SoccerGet for ball
    height, so the airborne/grounded split (`airborne_hang_time`) is a
    rough same-tick estimate re-derived from CURRENT speed every tick, not
    a fixed value anchored to the actual kick moment. As the ball
    decelerates mid-flight this re-guess shifts, which is why the glide
    segment can look unstable — that's a real sensing gap, not a drawing
    bug. A "just assume grounded" rewrite was tried and tested WORSE
    against the live champion (see `ball_physics.airborne_hang_time`'s
    docstring) — a proper fix needs to actually anchor hang-time to the
    real kick-release event, not drop airborne detection altogether.

    Returns the 4 breakpoints (`landing`, `leg1_end`, `leg2_end`,
    `leg3_end`) so `draw_ball_path_and_threats` can both draw them and
    reuse the final one as "the guaranteed interception point."
    """
    pred_vel = Vector3(ball_vel.x, Float(0), ball_vel.z)
    pred_speed = Magnitude(pred_vel)
    pred_hang = airborne_hang_time(pred_speed)
    landing, leg1, leg2, leg3, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end = trajectory_breakpoints(
        ball, pred_vel, pred_hang, ball.y
    )
    return pred_hang, leg1, leg2, leg3, landing, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end


def draw_trajectory_line(start, seg1_end, seg2_end, seg3_start, seg3_end, seg4_start, seg4_end, active, color="White", line_width=0.05, disc_radius=0.3, disc_opacity=0.2):
    """Draw 4 connected segments (start->seg1_end->seg2_end,
    seg3_start->seg3_end, seg4_start->seg4_end) plus an endpoint disc,
    only while `active`. Each segment's endpoints are expected to already
    collapse to a shared point once a further leg isn't reached (see
    `ball_physics.trajectory_breakpoints`), so this never needs its own
    has_wall bookkeeping — it just draws whatever it's given."""
    DebugDrawLine(ConditionalSetVector3(active, start, OFF_PITCH), ConditionalSetVector3(active, seg1_end, OFF_PITCH), Float(line_width), color)
    DebugDrawLine(ConditionalSetVector3(active, seg1_end, OFF_PITCH), ConditionalSetVector3(active, seg2_end, OFF_PITCH), Float(line_width), color)
    DebugDrawLine(ConditionalSetVector3(active, seg3_start, OFF_PITCH), ConditionalSetVector3(active, seg3_end, OFF_PITCH), Float(line_width), color)
    DebugDrawLine(ConditionalSetVector3(active, seg4_start, OFF_PITCH), ConditionalSetVector3(active, seg4_end, OFF_PITCH), Float(line_width), color)
    DebugDrawDisc(ConditionalSetVector3(active, seg4_end, OFF_PITCH), Float(disc_radius), Float(disc_opacity), color)


def draw_kick_preview(origin, aim_dir, charge, live_y, active, color="Purple") -> None:
    """Preview where a shot would land IF released right now, at the
    current charge, toward `aim_dir` — exact physics (known charge, known
    direction), not an estimate. Meant for a human-controlled rig
    (`probes/build_perfect_controller.py`): hold a charge, watch this line,
    release, and compare the line to where the ball actually ends up — the
    fastest way to find a real miscalibration in the trajectory model
    instead of guessing from a defensive-prediction failure after the fact.
    """
    from titanium.ball_physics import simulate_kick_breakpoints

    _hang, landing, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end = simulate_kick_breakpoints(
        origin, aim_dir, charge, live_y
    )
    draw_trajectory_line(origin, landing, leg1_end, leg2_start, leg2_end, leg3_start, leg3_end, active, color)


def draw_ball_path_and_threats(ball, ball_vel, loose, team4, opponents, opp_goal, opp_left_post, opp_right_post, team_goal, left_post, right_post):
    """Draws the always-on predicted ball path (truncated at whichever
    comes first: natural physics stop, or the earliest point any of the 8
    players could already reach it at full sprint — "where Titanium is
    trying to intercept it, unless someone gets there faster"), the
    threat-tree from that guaranteed-interception point (direct shot cone +
    pass-then-shoot to every teammate, mirrored at whichever goal the
    predicted receiver threatens), and the guaranteed-unopposed-walk-in
    flag in both directions.
    """
    pred_hang, pred_leg1, pred_leg2, pred_leg3, pred_landing, pred_leg1_end, pred_leg2_start, pred_leg2_end, pred_leg3_start, pred_leg3_end = predict_ball_path(ball, ball_vel, loose)

    all_players = list(team4) + list(opponents)

    def fastest_arrival(point):
        best = Distance(all_players[0], point) / Float(SPRINT_SPEED)
        for pl in all_players[1:]:
            t = Distance(pl, point) / Float(SPRINT_SPEED)
            best = ConditionalSetFloat(CompareFloats(t, best, "<"), t, best)
        return best

    t0 = pred_hang
    t1 = t0 + pred_leg1["event_time"]
    t2 = t1 + ConditionalSetFloat(pred_leg1["has_wall"], pred_leg2["event_time"], Float(0))
    t3 = t2 + ConditionalSetFloat(And(pred_leg1["has_wall"], pred_leg2["has_wall"]), pred_leg3["event_time"], Float(0))

    beat0 = CompareFloats(fastest_arrival(pred_landing), t0, "<=")
    beat1 = CompareFloats(fastest_arrival(pred_leg1_end), t1, "<=")
    beat2 = CompareFloats(fastest_arrival(pred_leg2_end), t2, "<=")
    beat3 = CompareFloats(fastest_arrival(pred_leg3_end), t3, "<=")

    keep1 = Not(beat0)
    keep2 = And(keep1, Not(beat1))
    keep3 = And(keep2, Not(beat2))

    # Collapse unreached legs to the BALL, never to OFF_PITCH. The sim resolves
    # both endpoints of a line identically so an OFF_PITCH pair is zero-length
    # and invisible, but Unity resolves the same condition inconsistently
    # across the two ends -- leaving one at the real point and the other at
    # (999,999), which renders as a line streaking off the map. Collapsing to
    # the ball means the worst case is a zero-length dot at the ball itself.
    seg1_end = ConditionalSetVector3(loose, pred_landing, ball)
    seg2_end = ConditionalSetVector3(And(loose, keep1), pred_leg1_end, seg1_end)
    seg3_start = ConditionalSetVector3(And(loose, keep1), pred_leg2_start, seg2_end)
    seg3_end = ConditionalSetVector3(And(loose, keep2), pred_leg2_end, seg2_end)
    seg4_start = ConditionalSetVector3(And(loose, keep2), pred_leg3_start, seg3_end)
    seg4_end = ConditionalSetVector3(And(loose, keep3), pred_leg3_end, seg3_end)

    # Always visible, not gated on `loose` -- a held ball still shows where it
    # would come to rest, which is what makes a wrong prediction obvious on screen.
    draw_trajectory_line(ball, seg1_end, seg2_end, seg3_start, seg3_end, seg4_start, seg4_end, Bool(True), "White", disc_radius=0.35, disc_opacity=0.25)

    # Threat tree from the guaranteed-interception point: whoever actually
    # reaches `seg4_end` first is treated as the next ball-holder, and we
    # draw the SAME direct-shot + pass-then-shoot tree from there as if they
    # already had it — "the ball is guaranteed to reach them, so their shot
    # cone (and their teammates' cones after a pass) is a real threat now,
    # not just once they're physically holding it." Green/pass-cyan when
    # WE'D be the ones receiving it, red/pass-cyan when they would.
    owner_pos = team4[0]
    owner_is_team = Bool(True)
    owner_t = Distance(team4[0], seg4_end) / Float(SPRINT_SPEED)
    for pl in team4[1:]:
        t = Distance(pl, seg4_end) / Float(SPRINT_SPEED)
        better = CompareFloats(t, owner_t, "<")
        owner_t = ConditionalSetFloat(better, t, owner_t)
        owner_pos = ConditionalSetVector3(better, pl, owner_pos)
        owner_is_team = ConditionalSetBool(better, Bool(True), owner_is_team)
    for pl in opponents:
        t = Distance(pl, seg4_end) / Float(SPRINT_SPEED)
        better = CompareFloats(t, owner_t, "<")
        owner_t = ConditionalSetFloat(better, t, owner_t)
        owner_pos = ConditionalSetVector3(better, pl, owner_pos)
        owner_is_team = ConditionalSetBool(better, Bool(False), owner_is_team)
    tree_active = loose  # only meaningful once there's a real predicted receiver
    we_receive = And(tree_active, owner_is_team)
    they_receive = And(tree_active, Not(owner_is_team))

    def draw_threat_from(origin, active, goal_pt, gl_post, gr_post, teammates, cone_color):
        _, tl = post_tangent(origin, gl_post, POST_AIM_RADIUS)
        _, tr = post_tangent(origin, gr_post, POST_AIM_RADIUS)
        DebugDrawLine(ConditionalSetVector3(active, origin, OFF_PITCH), ConditionalSetVector3(active, tl, OFF_PITCH), Float(0.04), cone_color)
        DebugDrawLine(ConditionalSetVector3(active, origin, OFF_PITCH), ConditionalSetVector3(active, tr, OFF_PITCH), Float(0.04), cone_color)
        for mate in teammates:
            DebugDrawLine(ConditionalSetVector3(active, origin, OFF_PITCH), ConditionalSetVector3(active, mate, OFF_PITCH), Float(0.03), "Cyan")
            _, mtl = post_tangent(mate, gl_post, POST_AIM_RADIUS)
            _, mtr = post_tangent(mate, gr_post, POST_AIM_RADIUS)
            DebugDrawLine(ConditionalSetVector3(active, mate, OFF_PITCH), ConditionalSetVector3(active, mtl, OFF_PITCH), Float(0.03), cone_color)
            DebugDrawLine(ConditionalSetVector3(active, mate, OFF_PITCH), ConditionalSetVector3(active, mtr, OFF_PITCH), Float(0.03), cone_color)

    # We're predicted to receive it: threat tree points at the ENEMY goal.
    draw_threat_from(owner_pos, we_receive, opp_goal, opp_left_post, opp_right_post, team4, "Green")
    # They're predicted to receive it: threat tree points at OUR goal.
    draw_threat_from(owner_pos, they_receive, team_goal, left_post, right_post, opponents, "Red")

    # Guaranteed unopposed walk-in: shortest path into the usable mouth
    # (closest point on the post–post goal-line segment), same geometry as
    # scoring eval — not the goal center. Race samples along that walk.
    def draw_unopposed_walk(active, origin, gl_post, gr_post, defenders, color):
        unopposed, goal_pt = unopposed_walk_in(origin, gl_post, gr_post, defenders)
        show = And(active, unopposed)
        DebugDrawLine(ConditionalSetVector3(show, origin, OFF_PITCH), ConditionalSetVector3(show, goal_pt, OFF_PITCH), Float(0.12), color)
        DebugDrawDisc(ConditionalSetVector3(show, goal_pt, OFF_PITCH), Float(0.5), Float(0.35), color)

    # Danger: they receive it and nobody on our side can stop the walk-in.
    draw_unopposed_walk(they_receive, owner_pos, left_post, right_post, team4, "Black")
    # Opportunity: we receive it and nobody on their side can stop it either.
    draw_unopposed_walk(we_receive, owner_pos, opp_left_post, opp_right_post, opponents, "Gray")


def plot_anti_tackle(slot: int, has_ball, debug: dict) -> None:
    """TimePlot anti-tackle diagnostics for post-game tackle forensics.

    FailReason (read after a match in TimePlot):
      0 — carrying and a safe heading was found (or predicted safe)
      1 — carrying but no probe found a safe ball angle this tick
      2 — lost the ball this tick after last tick predicted safe (miscalculation)
    """
    prefix = f"Titanium.P{slot}.AT"
    any_safe = debug["any_safe"]
    predicted_safe = debug["predicted_safe"]
    had_ball_key = f"{prefix}.HadBall"
    pred_key = f"{prefix}.PredSafe"

    was_carrying = CompareFloats(GetVariable(had_ball_key), Float(0.5), ">")
    pred_was_safe = CompareFloats(GetVariable(pred_key), Float(0.5), ">")
    opp_has = SoccerGetBool("Opponent Has Ball")
    miscalc = And(Not(has_ball), And(was_carrying, And(pred_was_safe, opp_has)))

    fail = Float(0)
    fail = ConditionalSetFloat(And(has_ball, Not(any_safe)), Float(1), fail)
    fail = ConditionalSetFloat(miscalc, Float(2), fail)

    pred_f = ConditionalSetFloat(predicted_safe, Float(1), Float(0))
    any_f = ConditionalSetFloat(any_safe, Float(1), Float(0))

    TimePlot(String(f"{prefix}.FailReason"), "Yellow", String(""), fail)
    TimePlot(String(f"{prefix}.PredictedSafe"), "Cyan", String(""), pred_f)
    TimePlot(String(f"{prefix}.AnySafeFound"), "Green", String(""), any_f)
    TimePlot(String(f"{prefix}.ChosenOffset"), "White", String(""), debug["chosen_offset"])
    if "pass_danger" in debug:
        TimePlot(
            String(f"{prefix}.PassDanger"),
            "Orange",
            String(""),
            ConditionalSetFloat(debug["pass_danger"], Float(1), Float(0)),
        )
    if "keep_penalty" in debug:
        TimePlot(String(f"{prefix}.KeepPenalty"), "Red", String(""), debug["keep_penalty"])
    if "best_eval" in debug:
        TimePlot(String(f"{prefix}.BestEval"), "Cyan", String(""), debug["best_eval"])
    if "need_pass" in debug:
        TimePlot(
            String(f"{prefix}.NeedPass"),
            "Orange",
            String(""),
            ConditionalSetFloat(debug["need_pass"], Float(1), Float(0)),
        )
    if "any_forward_next" in debug:
        TimePlot(
            String(f"{prefix}.FwdNext"),
            "Green",
            String(""),
            ConditionalSetFloat(debug["any_forward_next"], Float(1), Float(0)),
        )
    if "trapped_next" in debug:
        TimePlot(
            String(f"{prefix}.TrappedNext"),
            "Red",
            String(""),
            ConditionalSetFloat(debug["trapped_next"], Float(1), Float(0)),
        )

    SetVariable(had_ball_key, ConditionalSetFloat(has_ball, Float(1), Float(0)))
    SetVariable(pred_key, ConditionalSetFloat(has_ball, pred_f, Float(0)))

