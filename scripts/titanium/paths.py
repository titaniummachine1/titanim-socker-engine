"""Build/deploy file locations — I/O concern, not gameplay logic."""
from __future__ import annotations

import sys
from pathlib import Path

from titanium._env import ROOT

SAVES = (
    Path.home()
    / "AppData"
    / "LocalLow"
    / "Unicorn One"
    / "AIComp"
    / "Saves"
    / "Soccer"
    / "Titanium.txt"
)
# Built graph is also dropped into the sim repo so the drill / headless
# harnesses can load it. That path stays gitignored there for artifact hygiene,
# not secrecy (both repos are public): a graph is 5-16 MB of JSON rewritten by
# every build, so committing it is churn. The source lives here.
LOCAL_OUT = ROOT.parents[0] / "aicomp-soccer-sim" / "data" / "titanium" / "Titanium.txt"
# Every build lands here first. SAVES/LOCAL_OUT (the "live" submission copy
# and the sim's gated-champion copy) are only touched by --promote, after a
# candidate has actually won its gate — a build must never silently
# overwrite whatever's currently accepted as strongest.
CANDIDATE_OUT = ROOT / "out" / "Titanium_candidate.txt"
BACKUPS_DIR = ROOT.parents[0] / "aicomp-soccer-sim" / "data" / "titanium" / "backups"
# Every build also lands here, unconditionally, so a candidate can be loaded
# and watched live in-game immediately without waiting on a gate result —
# separate from SAVES (the real submission file), which only --promote touches.
SAVES_TEST = SAVES.parent / "Titanium_test.txt"
# Same bytes in the sim repo so watch_bots / headless can load without Unity path.
LOCAL_TEST = LOCAL_OUT.parent / "Titanium_test.txt"
# Challenger mirror under Unity Saves (pick "Titanium_release" in-game).
# For debug builds, use "Titanium_debug" instead.
if "--debug" in sys.argv or "--normal" in sys.argv:
    SAVES_CHALLENGER = SAVES.parent / "Titanium_debug.txt"
else:
    SAVES_CHALLENGER = SAVES.parent / "Titanium_release.txt"
