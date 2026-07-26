"""MatchProbe — hold the ball, pace up and down, and log the ENTIRE Soccer API.

Run this as HOME against `blank_probe.py` as AWAY. The blank side never presses
Interact, so P1 takes the ball at kickoff and keeps it: Interact is held true
forever, which charges a shot but never releases it, so neither side can score.
The match therefore cannot end on goals and has to run to whatever natural end
the game has — the run needed to answer:

  * how long is a full match, in the recording's own seconds?
  * does the game remove players as it goes, and if so at what times?
  * how is a goalless match settled — possession?
  * which API getters are actually LIVE, and which are dead stubs?

What it logs
------------
Everything the Soccer API exposes, not just positions:

    SoccerGetFloat      29 channels   score, shots, possession %, attacking %,
                                      stamina per player, shot charges, ball
                                      speed, interact radius, field geometry,
                                      distance to nearest opponent per player
    SoccerGetBool       49 channels   has-ball per player, nearby, closest,
                                      open, kickoff flags, winning, ball side,
                                      ball headed at goal, home/away
    SoccerGetTransform  27 x 2        every player, ball, goals, posts, and the
                                      "nearest X to Y" resolvers
    SoccerGetVector3    55 x 3        opt-in via --vectors: velocities, clear
                                      directions, landmarks, open-player picks
                                      (plus an IsNull channel each, since
                                      several of these legitimately return null)

That is ~130 channels by default and ~300 with --vectors. TimePlot load is real
— Titanium's own debug module keeps its channel count down specifically to
avoid lagging the live game — so if the game stutters, re-run with --minimal
(floats + bools only, 78 channels).

    python scripts\\blank_probe.py            # once: the control opponent
    python scripts\\circle_probe.py           # floats + bools + transforms
    python scripts\\circle_probe.py --vectors # everything
    python scripts\\circle_probe.py --minimal # floats + bools only
    # AIComp Soccer: HOME = MatchProbe, AWAY = Blank. Play to the very end.
    # F1 -> export the timeplot JSON.
    python scripts\\analyze_clock_probe.py <json> --final-minute <clock at whistle>

The clock is read DIRECTLY. `SoccerGetFloat` has 44 options, not 29 — the last
six are `Current Simulation Time`, `Max Simulation Time`, `Simulation Time
Remaining`, `Delta Time`, `Fixed Delta Time` and `Pi`. So match length and
elapsed time need no counting or correlation at all.

Read them from the REAL GAME only. The offline sim's float catalog is
misaligned with the dropdown indices past ~37: asked for those six labels it
returns 1.0 / 0.0 / 180.0 / 180.0 / 0.019 / 0.019, i.e. `Pi` comes back as
0.019. Whatever the sim reports for this range is meaningless.

The TimePlot x axis (game seconds, one sample per tick) remains a useful
cross-check: `dt = span / (samples - 1)` should agree with `Fixed Delta Time`.

Motion: straight up and down Z, turning at +-Z_LIMIT, X never changed. At x ~ 0
the carrier stays ~40 m from both goals and cannot score on himself. The target
is aimed well past the turnaround so it is never reached — an unreachable
target is what keeps him walking instead of arriving and stuttering.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.join(_HERE, "..", "..", "worldcupteams")
sys.path.insert(0, os.path.join(_WT, "AIGamePyLibrary"))
sys.path.insert(0, _WT)

from AIGamePyLibrary import *  # noqa: E402
from AIGamePyLibrary.data import DROPDOWN_OPTIONS  # noqa: E402
import ball_trajectory_graph as _traj  # noqa: E402,F401  installs Node +/-/* overloads

SOCCER_SAVES = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    "AppData", "LocalLow", "Unicorn One", "AIComp", "Saves", "Soccer",
)

RAMP_VAR = "_MatchProbeRamp"
KICKOFF_VAR = "_MatchProbeKickoffs"
WAS_KICKOFF_VAR = "_MatchProbeWasKickoff"
DIR_VAR = "_MatchProbeDir"

# Turn around HERE, aim THERE. Both are inside the +-25 pitch, so the carrier
# never reaches the target and never pins against a wall — an unreachable
# target is what keeps him walking, but only if it is still on the field.
# Aiming at +-35 walked him into the top boundary and he stuck there.
# Interact pulse: ramp units per press/release cycle. Kept short so a dropped
# ball is re-claimed within a fraction of a second.
PULSE_PERIOD = 12.0
Z_TURN = 15.0
Z_AIM = 22.0
# Ramp ticks per full up-down cycle. The ramp rises once per tick (52.63/s
# measured), so 420 is a ~4 s half-stroke — long enough to cover ground, short
# enough that a stall is obvious on the plot.
PACE_PERIOD = 840.0
# Distinct parking spots per slot. They MUST differ: when every idle player sat
# on the same point, a player the game removed was indistinguishable from one
# standing still, which is exactly why the 3v3 removal was invisible in the
# 2026-07-26 14:23 recording.
PARK = {1: (26.0, -14.0), 2: (26.0, 14.0), 3: (32.0, -7.0), 4: (32.0, 7.0)}
# Stand-in for a null Vector3, far outside the +-40 x +-25 pitch so it can never
# be confused with a real reading.
NULL_SENTINEL = -999.0

WITH_VECTORS = "--vectors" in sys.argv
MINIMAL = "--minimal" in sys.argv
# --presence: the smallest set that can answer "does the game remove players,
# and how does a goalless match end". 151 channels x 3908 samples CRASHED the
# real game on export, so a run that has to survive a full match needs to be
# far leaner than the full sweep. ~26 channels.
PRESENCE = "--presence" in sys.argv
PRESENCE_FLOATS = (
    "Current Simulation Time", "Simulation Time Remaining", "Max Simulation Time",
    "Team Score", "Opponent Score",
    "Team Possession %", "Opponent Possession %",
)
PRESENCE_BOOLS = ("Is Active Graph", "Is Kickoff")
# Independent of position: a removed player's distance-to-nearest-opponent
# should behave differently from a parked one. Two unrelated signals beat one.
# Trimmed to the bare minimum: 51 channels x 9474 samples nearly crashed the
# real game's export. Positions alone identify a removal; the stamina and
# distance corroborators cost 8 more channels than the run can afford.
PRESENCE_FLOAT_PREFIXES = ()


def _short(label: str) -> str:
    """Compact channel name — TimePlot legends are narrow."""
    out = label
    for long, short in (
        ("Opponent Player ", "O"), ("Team Player ", "T"),
        ("Opponent ", "Opp"), ("Teammate ", "Mate"), ("Team ", "Tm"),
        ("Direction of ", "Dir"), ("Clear direction from ", "Clear"),
        ("Distance from ", "Dist"), (" to nearest ", "->"),
        ("Is Ball Nearby ", "Near"), ("Is ", ""), (" Closest Teammate to Ball", "Closest"),
    ):
        out = out.replace(long, short)
    return "".join(ch for ch in out if ch.isalnum() or ch in "->_")


def plot(name: str, color: str, value) -> None:
    TimePlot(String(f"P.{name}"), color, String(""), value)


def plot_bool(name: str, color: str, flag) -> None:
    plot(name, color, ConditionalSetFloat(flag, Float(1), Float(0)))


def plot_xz(name: str, color: str, vec) -> None:
    parts = Vector3Split(vec)
    plot(f"{name}.X", color, parts.x)
    plot(f"{name}.Z", color, parts.z)


def main() -> None:
    side = ConditionalSetFloat(SoccerGetBool("Is Home Team"), Float(-1), Float(1))

    def spot(x, z):
        return ScaleVector3(Vector3(Float(x), Float(0), Float(z)), side)

    # P1 on the centre spot so it is already on the ball at kickoff. Faceoff
    # coordinates are WORLD ABSOLUTE, so every spot is mirrored off `Is Home
    # Team` exactly as `titanium.graph._faceoff_spots` does; the centre spot is
    # its own mirror. The other three sit deep and wide, out of the way.
    InitializeSoccer(
        "MatchProbe", "Poland",
        spot(0.0, 0.0), spot(20.0, -10.0), spot(20.0, 10.0), spot(34.0, 0.0),
    )

    # --- tick ramp (plot-side time axis only) -----------------------------
    # The sum feeds SetVariable and nothing else; readers use the plain
    # GetVariable, because reads are idempotent while an accumulator is not.
    ramp = GetVariable(RAMP_VAR)
    SetVariable(RAMP_VAR, ramp + Float(1))

    def pacing_for(slot: int):
        """Up-and-down target for `slot`, turning on its OWN z.

        Position-based, not timer-based. The ramp cannot be used for timing:
        it rose 52.63/s in one build and 72.49/s in the next, because an
        increment executes once per CONSUMER, so its rate depends on graph
        shape. Position has no such problem, and the transform read is world
        scale (a carrier reached z = 24.01, i.e. the real pitch edge).

        Hysteresis via a stored per-slot direction, so the flip happens at
        +-Z_TURN and not repeatedly around one threshold.
        """
        var = f"{DIR_VAR}{slot}"
        me_z = Vector3Split(
            RelativePosition(SoccerGetTransform(f"Team Player {slot}"), "Self")).z
        going_up = CompareFloats(GetVariable(var), Float(0), ">=")  # 0 => start up
        at_top = CompareFloats(me_z, Float(Z_TURN), ">=")
        at_bottom = CompareFloats(me_z, Float(0) - Float(Z_TURN), "<=")
        new_up = ConditionalSetBool(
            at_top, Bool(False), ConditionalSetBool(at_bottom, Bool(True), going_up))
        # A flag, not an accumulator: rewriting it twice in a tick is harmless.
        SetVariable(var, ConditionalSetFloat(new_up, Float(1), Float(0) - Float(1)))
        return ConditionalSetVector3(
            new_up,
            Vector3(Float(0), Float(0), Float(Z_AIM)),
            Vector3(Float(0), Float(0), Float(0) - Float(Z_AIM)),
        )

    # --- restart counter (rising edge) ------------------------------------
    is_kickoff = SoccerGetBool("Is Kickoff")
    was_kickoff = CompareFloats(GetVariable(WAS_KICKOFF_VAR), Float(0.5), ">")
    kickoffs = GetVariable(KICKOFF_VAR)
    SetVariable(KICKOFF_VAR, kickoffs + ConditionalSetFloat(
        And(is_kickoff, Not(was_kickoff)), Float(1), Float(0)))
    SetVariable(WAS_KICKOFF_VAR, ConditionalSetFloat(is_kickoff, Float(1), Float(0)))

    plot("Ramp", "Gray", ramp)
    plot("Kickoffs", "Orange", kickoffs)

    counts = {"float": 0, "bool": 0, "transform": 0, "vector": 0}

    # Anything reconstructible from the positions we already log is dead weight
    # here — it costs TimePlot bandwidth and tells us nothing new. Dropped for
    # that reason: every "Nearest X to Y" resolver, "Is X Closest to Ball",
    # "Ball On Team/Opponent Side", "Distance from X to nearest Opponent", the
    # fixed landmarks (corners / midfield / centre), and the "Direction of
    # <thing> from <player>" family. All of those are arithmetic on positions.
    #
    # What stays is state the engine owns and we cannot derive:
    #   * score / shots / possession % / attacking %  — and note that the two
    #     percentages are ratios over ELAPSED MATCH TIME, so together with the
    #     tick count they expose the game's own notion of how far through the
    #     match it is. That is the clock question, answered indirectly.
    #   * stamina and shot charge per player — internal, invisible otherwise
    #   * "Is X Open" and the "clear direction" family — the engine's own
    #     lane/marking judgement, not ours
    #   * "Is Ball Headed Towards Goal" — an engine-side prediction
    #   * "Is Active Graph" — meaning unknown, and a plausible signal for a
    #     player being taken out of play, which is the whole point of this run
    #   * field geometry constants, logged so the real numbers are on record
    skip_all = "--all" in sys.argv

    def wanted_float(label: str) -> bool:
        if PRESENCE:
            return (label in PRESENCE_FLOATS
                    or label.startswith(PRESENCE_FLOAT_PREFIXES))
        return skip_all or "to nearest Opponent" not in label

    def wanted_bool(label: str) -> bool:
        if PRESENCE:
            return label in PRESENCE_BOOLS
        if skip_all:
            return True
        # "Is X Open" is a KNOWN rule, not a mystery: it is true when no
        # opponent is within 2x the interact radius of that player. It is
        # direction-independent — a marker on the far side counts exactly as
        # much as one in the passing lane — so it is pure arithmetic on
        # positions and the interact radius, both of which are already logged.
        return not ("Closest" in label or "Ball On " in label or " Open" in label)

    def wanted_transform(label: str) -> bool:
        if PRESENCE:
            # Only the eight players — their positions ARE the removal evidence.
            return "Player" in label
        if skip_all:
            return True
        return "Nearest" not in label

    def wanted_vector(label: str) -> bool:
        if skip_all:
            return True
        if label.startswith("Direction of"):
            return False
        for landmark in ("Corner", "Midfield", "Center Field"):
            if landmark in label:
                return False
        return True

    for label in DROPDOWN_OPTIONS["SoccerGetFloat"]:
        if wanted_float(label):
            plot(f"F.{_short(label)}", "Green", SoccerGetFloat(label))
            counts["float"] += 1

    for label in DROPDOWN_OPTIONS["SoccerGetBool"]:
        if wanted_bool(label):
            plot_bool(f"B.{_short(label)}", "Cyan", SoccerGetBool(label))
            counts["bool"] += 1

    if not MINIMAL or PRESENCE:
        for label in DROPDOWN_OPTIONS["SoccerGetTransform"]:
            if wanted_transform(label):
                pos = RelativePosition(SoccerGetTransform(label), "Self")
                plot_xz(f"X.{_short(label)}", "Yellow", pos)
                counts["transform"] += 1

    # Several Vector3 getters legitimately return null (a clear direction when
    # no lane exists). A null would plot as an indistinguishable 0, so each gets
    # an explicit IsNull channel and an off-pitch sentinel.
    if WITH_VECTORS:
        for label in DROPDOWN_OPTIONS["SoccerGetVector3"]:
            if not wanted_vector(label):
                continue
            raw = SoccerGetVector3(label)
            is_null = IsNull(raw)
            safe = ConditionalSetVector3(
                is_null, Vector3(Float(NULL_SENTINEL), Float(0), Float(NULL_SENTINEL)), raw)
            name = _short(label)
            plot_xz(f"V.{name}", "Magenta", safe)
            plot_bool(f"V.{name}.Null", "Red", is_null)
            counts["vector"] += 1

    # Offline readout for probe_dump (TimePlot is a null stub headless).
    DebugDrawDisc(Vector3(ramp, Float(0), Float(0)), Float(0.1), Float(0.1), "White")

    # --- controllers: WHOEVER holds the ball paces --------------------------
    # The carrier is assigned dynamically, not hardcoded to P1. The game removes
    # players as the match runs (confirmed: the scoreboard shows 3v3), so a
    # fixed carrier gets deleted mid-match, nobody herds the ball, and the
    # stale-ball whistle starts firing.
    #
    # Everyone holds Interact permanently: it charges a shot but never releases
    # (release happens on the falling edge), so the ball can never be kicked and
    # no goal can be scored by either side — and any player adjacent to a loose
    # ball claims it automatically. That is what makes the handover work without
    # needing the ball's position in an ambiguous coordinate frame.
    #
    # Parking spots are LITERAL world coordinates, never the player's own
    # read-back position: `RelativePosition(t, "Self")` is relative to the
    # evaluating player, so "move_to = my own position" is ~(0,0) — the centre
    # spot. Every idle player walked there and stopped, on both teams, which is
    # the bug that masked the removals.
    # Ball position reads at world scale (an earlier export ranged -40.64..18.95).
    ball = RelativePosition(SoccerGetTransform("Ball"), "Self")

    for slot in (1, 2, 3, 4):
        mine = SoccerGetBool(f"Team Player {slot} Has Ball")
        # SOMEBODY must go and get it. Without this the ball is never claimed:
        # every player either paces (needs the ball) or parks, so a kickoff
        # leaves it sitting on the centre spot until the stale-ball whistle
        # fires, which restarts the kickoff, which whistles again — a loop that
        # never plays a single second of football.
        #
        # The fetcher is chosen dynamically by the engine's own "closest
        # teammate" flag rather than hardcoded, so it survives the removals:
        # whoever is left and nearest goes, and exactly one player goes.
        fetch = SoccerGetBool(f"Is Team Player {slot} Closest Teammate to Ball")
        px, pz = PARK[slot]
        target = ConditionalSetVector3(fetch, ball, spot(px, pz))
        target = ConditionalSetVector3(mine, pacing_for(slot), target)
        # Interact is an IMPULSE, not a held advantage (confirmed by the game's
        # author). Holding it only helps to CHARGE a shot; a claim or a tackle
        # fires once on the press and needs a RELEASE before it can fire again.
        # Holding it permanently therefore claims the ball exactly once — lose
        # it and the player can never take it back.
        #
        # So: hold while carrying (that is the charge, and it is the one case
        # holding helps), and otherwise pulse, giving a fresh rising edge a few
        # times a second. Duty comes from the ramp because reads are idempotent;
        # a flag that inverted itself would be corrupted by the repeated writes
        # this graph layer is prone to.
        pulse = CompareFloats(Modulo(ramp, Float(PULSE_PERIOD)),
                              Float(PULSE_PERIOD / 2.0), "<")
        SoccerController(slot, target, Bool(False), Or(mine, pulse))

    out = os.path.join(SOCCER_SAVES, "MatchProbe.txt")
    os.makedirs(SOCCER_SAVES, exist_ok=True)
    SaveData(out, "grid")

    channels = (counts["float"] + counts["bool"] + counts["transform"] * 2
                + counts["vector"] * 3 + 2)
    print(f"Wrote {out}")
    print(f"  floats {counts['float']}  bools {counts['bool']}  "
          f"transforms {counts['transform']}  vector3s {counts['vector']}")
    print(f"  ~{channels} TimePlot channels")
    if not WITH_VECTORS:
        print("  (add --vectors for the 55 Vector3 getters; --minimal to drop transforms)")
    print(
        "\nRun it:\n"
        "  1. AIComp Soccer -> HOME = MatchProbe, AWAY = Blank\n"
        "  2. Play to the very end. Note the final on-screen clock.\n"
        "  3. F1 -> export the timeplot JSON.\n"
        "  4. python scripts\\analyze_clock_probe.py <json> --final-minute <N>\n"
    )


if __name__ == "__main__":
    main()
