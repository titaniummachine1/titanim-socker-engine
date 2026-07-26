"""ClockProbe — work out what an in-game minute actually is.

The Soccer API has NO clock. `SoccerGetFloat` exposes score, shots, possession
%, stamina and field geometry — nothing temporal — and the library's
`Game.DeltaTime` is a `VolleyballGetFloat`, invalid in a Soccer graph. So the
graph cannot read match time; it has to count its own ticks and let the
recording supply the time axis.

That is exactly what makes this measurable. TimePlot's x axis is the game's own
seconds, so plotting a raw tick counter against x gives the tick rate directly:

    ticks per second = slope of ClockProbe.Ticks vs x
    seconds per tick = 1 / that            (compare to the sim's 0.019)

Run it, let the match play to its NATURAL end (do not stop it early — the final
sample is the whole point: it tells you the real match length in both ticks and
seconds), then note the final on-screen match clock. `analyze_clock_probe.py`
turns the exported JSON plus that one number into minutes-per-second.

    python scripts\clock_probe.py
    # in AIComp Soccer, load "ClockProbe" as Home, play a FULL match,
    # then F1 -> export the timeplot JSON.
    python scripts\analyze_clock_probe.py <timeplot.json> --final-minute 90

Channels
--------
  Ticks            raw graph ticks since load; the primary measurement
  SecondsAt19ms    Ticks * 0.019 — predicted seconds IF dt matches the sim.
                   If this tracks x, the sim's FIXED_DT is right; if it drifts,
                   the ratio of the two IS the correction.
  MinutesIfSecond  Ticks * 0.019 / 60 — what the clock would read if a game
                   minute were a real minute. Expect this to sit far BELOW the
                   on-screen clock if minutes really tick like seconds.
  Score/Possession context so restarts and goals are visible on the same axis.
  Kickoffs         counts restarts; possession % is measured over play time,
                   not wall time, so restarts matter when interpreting it.

Everyone stands still on purpose: an idle match usually reaches full time
instead of ending early on the mercy rule, which is what makes the final sample
a clean read of match LENGTH.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
_WT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "worldcupteams")
sys.path.insert(0, os.path.join(_WT, "AIGamePyLibrary"))
sys.path.insert(0, _WT)

from AIGamePyLibrary import *  # noqa: E402
import ball_trajectory_graph as _traj  # noqa: E402,F401  installs Node +/-/* overloads

SOCCER_SAVES = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    "AppData", "LocalLow", "Unicorn One", "AIComp", "Saves", "Soccer",
)

# The sim's FIXED_DT. Used only to derive a PREDICTED seconds channel — if the
# prediction diverges from the recording's x axis, the sim constant is wrong,
# which is itself the finding.
SIM_FIXED_DT = 0.019

TICKS_VAR = "_ClockProbeTicks"
KICKOFF_VAR = "_ClockProbeKickoffs"
WAS_KICKOFF_VAR = "_ClockProbeWasKickoff"


def plot(name: str, color: str, value) -> None:
    TimePlot(String(f"ClockProbe.{name}"), color, String(""), value)


def main() -> None:
    InitializeSoccer(
        "ClockProbe", "Poland",
        Vector3(Float(-10), Float(0), Float(0)),
        Vector3(Float(-14), Float(0), Float(-6)),
        Vector3(Float(-14), Float(0), Float(6)),
        Vector3(Float(-30), Float(0), Float(0)),
    )

    # --- tick counter -----------------------------------------------------
    # MEASURED TRAP: the `prev + 1` expression must have EXACTLY ONE consumer.
    # Evaluation is demand-driven, so an increment feeding N consumers runs N
    # times per tick — with 8 consumers this counter read 8*t+1 instead of t
    # (probe_dump: 9, 17, 25, 41, 81, 161 for t = 1, 2, 3, 5, 10, 20).
    #
    # So: the sum feeds SetVariable and nothing else, and everything that
    # DISPLAYS the count reads `ticks` (a plain GetVariable). Reads are
    # idempotent, so any number of consumers is safe. The displayed value lags
    # the write by one tick, which is a constant offset and does not affect any
    # rate derived from it.
    ticks = GetVariable(TICKS_VAR)
    SetVariable(TICKS_VAR, ticks + Float(1))

    seconds_at_19ms = ticks * Float(SIM_FIXED_DT)

    # --- kickoff / restart counter ---------------------------------------
    # Rising edge only, so a kickoff that spans many ticks counts once.
    is_kickoff = SoccerGetBool("Is Kickoff")
    was_kickoff = CompareFloats(GetVariable(WAS_KICKOFF_VAR), Float(0.5), ">")
    rising = And(is_kickoff, Not(was_kickoff))
    kickoffs = GetVariable(KICKOFF_VAR) + ConditionalSetFloat(rising, Float(1), Float(0))
    SetVariable(KICKOFF_VAR, kickoffs)
    SetVariable(WAS_KICKOFF_VAR, ConditionalSetFloat(is_kickoff, Float(1), Float(0)))

    # --- channels ---------------------------------------------------------
    plot("Ticks", "White", ticks)
    plot("SecondsAt19ms", "Cyan", seconds_at_19ms)
    plot("MinutesIfRealMinute", "Magenta", seconds_at_19ms / Float(60))
    plot("Kickoffs", "Orange", kickoffs)
    plot("IsKickoff", "Yellow", ConditionalSetFloat(is_kickoff, Float(1), Float(0)))
    plot("TeamScore", "Green", SoccerGetFloat("Team Score"))
    plot("OppScore", "Red", SoccerGetFloat("Opponent Score"))
    plot("TeamPossessionPct", "Blue", SoccerGetFloat("Team Possession %"))
    plot("OppPossessionPct", "Gray", SoccerGetFloat("Opponent Possession %"))

    # Offline readout: probe_dump prints DebugDrawDisc even headless, where
    # TimePlot compiles to a null stub. X carries the tick count.
    DebugDrawDisc(Vector3(ticks, Float(0), Float(0)), Float(0.1), Float(0.1), "White")

    # Stand still: an idle match runs to full time instead of tripping the
    # mercy rule, so the last sample reads true match LENGTH.
    for slot, spot in (
        (1, Vector3(Float(-10), Float(0), Float(0))),
        (2, Vector3(Float(-14), Float(0), Float(-6))),
        (3, Vector3(Float(-14), Float(0), Float(6))),
        (4, Vector3(Float(-30), Float(0), Float(0))),
    ):
        SoccerController(slot, spot, Bool(False), Bool(False))

    out = os.path.join(SOCCER_SAVES, "ClockProbe.txt")
    os.makedirs(SOCCER_SAVES, exist_ok=True)
    SaveData(out, "grid")
    print(f"Wrote {out}")
    print(
        "\nNext:\n"
        "  1. AIComp Soccer -> load 'ClockProbe' as Home (any opponent).\n"
        "  2. Play a FULL match, to its natural end. Do not stop it early.\n"
        "  3. Note the FINAL on-screen match clock (the minute reading).\n"
        "  4. F1 -> export the timeplot JSON.\n"
        "  5. python scripts\\analyze_clock_probe.py <json> --final-minute <N>\n"
    )


if __name__ == "__main__":
    main()
