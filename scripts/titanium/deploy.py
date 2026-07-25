"""Candidate/test/promote file I/O — deliberately separate from
`titanium.graph`, which only builds the node graph in memory and knows
nothing about where any of this ends up on disk."""
from __future__ import annotations

import sys

from titanium._env import graph_data
from AIGamePyLibrary import SaveData
from titanium.paths import BACKUPS_DIR, CANDIDATE_OUT, LOCAL_OUT, SAVES, SAVES_TEST


def write_candidate() -> None:
    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SaveData(str(CANDIDATE_OUT), layout="grid")
    print(f"Wrote candidate: {CANDIDATE_OUT}")
    print(f"nodes={len(graph_data['serializableNodes'])} conns={len(graph_data['serializableConnections'])}")

    SAVES_TEST.parent.mkdir(parents=True, exist_ok=True)
    SAVES_TEST.write_text(CANDIDATE_OUT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote test build (load '{SAVES_TEST.stem}' in-game to try it yourself): {SAVES_TEST}")


def promote_if_requested() -> None:
    if "--promote" not in sys.argv:
        print(
            "\nNOT deployed to live — this only builds a candidate.\n"
            "Gate it first (aicomp-soccer-sim/scripts/gate_round_robin.py) "
            "against the currently-live build, then re-run with --promote "
            "once it's actually won."
        )
        return

    text = CANDIDATE_OUT.read_text(encoding="utf-8")
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_OUT.is_file():
        prior = LOCAL_OUT.read_bytes()
        n = 0
        backup_path = BACKUPS_DIR / "Titanium_pre_promote.txt"
        while backup_path.is_file() and backup_path.read_bytes() != prior:
            n += 1
            backup_path = BACKUPS_DIR / f"Titanium_pre_promote_{n}.txt"
        if not backup_path.is_file():
            backup_path.write_bytes(prior)
        print(f"Backed up previously-live build to {backup_path}")

    SAVES.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAVES.write_text(text, encoding="utf-8")
    LOCAL_OUT.write_text(text, encoding="utf-8")
    print(f"PROMOTED to live: {SAVES}")
    print(f"PROMOTED to live: {LOCAL_OUT}")
