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

Z_LIMIT = 15.0
OVERSHOOT = 20.0
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

    ball = RelativePosition(SoccerGetTransform("Ball"), "Self")
    me = RelativePosition(SoccerGetTransform("Team Player 1"), "Self")
    has_ball = SoccerGetBool("Team Player 1 Has Ball")

    # --- pace up and down, turning at +-Z_LIMIT ---------------------------
    # Bang-bang on a stored direction, with the turnaround driven by the
    # carrier's own position rather than a timer, so it self-corrects after a
    # bump. The stored value is a flag, not an accumulator, so a repeated write
    # in one tick is harmless.
    going_up = CompareFloats(GetVariable(DIR_VAR), Float(0), ">=")
    at_top = CompareFloats(me.z, Float(Z_LIMIT), ">=")
    at_bottom = CompareFloats(me.z, Float(0) - Float(Z_LIMIT), "<=")
    new_up = ConditionalSetBool(at_top, Bool(False),
                                ConditionalSetBool(at_bottom, Bool(True), going_up))
    SetVariable(DIR_VAR, ConditionalSetFloat(new_up, Float(1), Float(0) - Float(1)))
    target_z = ConditionalSetFloat(
        new_up, Float(Z_LIMIT + OVERSHOOT), Float(0) - Float(Z_LIMIT + OVERSHOOT))
    pacing = Vector3(me.x, Float(0), target_z)
    p1_target = ConditionalSetVector3(has_ball, pacing, ball)

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
            return label in PRESENCE_FLOATS
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

    # P1 carries; the rest hold station so they cannot steal off their own
    # carrier or crowd the lane.
    SoccerController(1, p1_target, Bool(False), Bool(True))
    for slot in (2, 3, 4):
        here = RelativePosition(SoccerGetTransform(f"Team Player {slot}"), "Self")
        SoccerController(slot, here, Bool(False), Bool(False))

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
