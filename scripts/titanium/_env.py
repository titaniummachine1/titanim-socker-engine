"""Shared foundation every other titanium module builds on.

Import this FIRST (or via `from titanium._env import *`) before anything
else in the package — it does the sys.path setup AIGamePyLibrary and
ball_trajectory_graph need to be importable at all, and importing
ball_trajectory_graph has the side effect of installing the Node
`+`/`-`/`*` operator overloads every other module relies on.
"""
from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]  # .../titanim-socker-engine/scripts
_ENGINE_ROOT = _SCRIPTS_ROOT.parent  # .../titanim-socker-engine
_WORKSPACE_ROOT = _ENGINE_ROOT.parent  # .../worldcup

for _p in (
    _WORKSPACE_ROOT / "worldcupteams" / "AIGamePyLibrary",
    _WORKSPACE_ROOT / "worldcupteams",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from AIGamePyLibrary import *  # noqa: E402,F401,F403
from AIGamePyLibrary.lib import data as graph_data  # noqa: E402,F401
import ball_trajectory_graph as traj  # noqa: E402,F401  side effect: installs Node +/-/* operators

# Titanium uses traj's small scalar helpers (StopDistance, TimeForDistance,
# SpeedAtDistance, AxisDistance -- 36 nodes) and its `_event_leg` physics, but
# never reads TrajectorySolve's outputs. Building it anyway cost 405 nodes and
# 29 SetVariable writes that nothing ever read: measured 5004 -> 4599 nodes and
# 11.78 -> 10.77 MB, with match results unchanged (Poponeta 16:4 either way).
traj.BUILD_TRAJECTORY_SOLVE = False

ROOT = _ENGINE_ROOT
