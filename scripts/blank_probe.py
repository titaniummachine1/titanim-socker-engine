"""Blank — a graph that does nothing at all. Load this as the OPPONENT.

The point of a control is that anything the recording shows is then the GAME's
doing, not the opponent's. This graph:

  * never presses Interact, so it can never take the ball, tackle, or shoot
  * commands every player to stand on its own current position, so nobody
    chases, drifts, or contests anything
  * writes no variables and emits no TimePlot/DebugDraw of its own, so it
    cannot collide with the probe's channels or add per-tick work

Paired with `circle_probe.py` on the other side, the carrier keeps the ball
unopposed for the whole match, no goals are scored, and the match has to run to
its natural end — which is precisely the run needed to see whether the game
removes players over time, and how it settles a goalless draw.

    python scripts\\blank_probe.py
    # AIComp Soccer -> load "Blank" as the AWAY team.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.join(_HERE, "..", "..", "worldcupteams")
sys.path.insert(0, os.path.join(_WT, "AIGamePyLibrary"))
sys.path.insert(0, _WT)

from AIGamePyLibrary import *  # noqa: E402
import ball_trajectory_graph as _traj  # noqa: E402,F401  installs Node +/-/* overloads

SOCCER_SAVES = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    "AppData", "LocalLow", "Unicorn One", "AIComp", "Saves", "Soccer",
)


def main() -> None:
    # Spawn deep in our own half and spread out, so the blank side is nowhere
    # near the centre circle and cannot accidentally block the carrier.
    spawns = [
        Vector3(Float(30), Float(0), Float(0)),
        Vector3(Float(34), Float(0), Float(-8)),
        Vector3(Float(34), Float(0), Float(8)),
        Vector3(Float(38), Float(0), Float(0)),
    ]
    InitializeSoccer("Blank", "Poland", *spawns)

    for slot in (1, 2, 3, 4):
        here = RelativePosition(SoccerGetTransform(f"Team Player {slot}"), "Self")
        # move_to = own position => zero desired velocity. Never sprint, never
        # interact.
        SoccerController(slot, here, Bool(False), Bool(False))

    out = os.path.join(SOCCER_SAVES, "Blank.txt")
    os.makedirs(SOCCER_SAVES, exist_ok=True)
    SaveData(out, "grid")
    print(f"Wrote {out}")
    print("Load 'Blank' as the AWAY team, 'CircleProbe' as HOME.")


if __name__ == "__main__":
    main()
