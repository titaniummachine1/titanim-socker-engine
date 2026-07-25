"""Build/deploy file locations — I/O concern, not gameplay logic."""
from __future__ import annotations

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
# Built graph is also dropped into the public sim repo purely so the drill /
# headless harnesses can load it. That path is gitignored there — the graph
# is a build artifact, the source stays here.
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
